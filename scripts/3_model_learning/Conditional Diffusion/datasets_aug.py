import os
import re
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import butter, filtfilt
from torchvision import transforms
import torchvision.transforms.functional as TF

# ===== User config imports (paths, counts, horizons, etc.) ===== #
from config import *  # expects: images_dir, csv_dir, demon_num, prediction_horizon, action_prediction, action_horizon

# =============================================================== #
# This version simplifies augmentation as requested:
# - Images: only brightness (illumination) change, small random factor
# - Position (x, y): add Gaussian noise
# - Force (Fz): add Gaussian noise, then clamp to <= 0 (physical constraint)
# - Set `augment_repeats=50` to get 50x augmented length
# - Augmentations happen BEFORE normalization; then global min/max normalization
# =============================================================== #

# ---------- Base image transform (after brightness) ---------- #
_base_img_transform = transforms.Compose([
    transforms.ToTensor(),            # (H,W,C)->(C,H,W), 0..255->0..1
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------- Utility: Low-pass filter for raw Fz preprocessing ---------- #
def lowpass_filter(x: np.ndarray, wn: float = 0.12, order: int = 2) -> np.ndarray:
    """Simple Butterworth low-pass with filtfilt for raw Fz preprocessing.
    Args:
        x: 1D array
        wn: normalized cutoff (0<wn<1). If you think in Hz, set wn = fc/(fs/2)
        order: filter order
    Returns: filtered 1D array
    """
    wn = max(1e-4, min(float(wn), 0.99))
    b, a = butter(order, wn, btype='low', analog=False)
    return filtfilt(b, a, x, method='gust')

# ---------- Augmenter (shared params per-sample) ---------- #
class Augmenter:
    def __init__(self, rng: random.Random,
                 img_brightness_delta: float = 0.15,
                 pos_noise_sigma: float = 0.2,
                 fz_noise_sigma: float = 0.2):
        self.rng = rng
        # Image brightness factor: 1 + U(-d, d)
        self.brightness = 1.0 + self.rng.uniform(-img_brightness_delta, img_brightness_delta)
        # Sequence noise std
        self.pos_noise_sigma = float(pos_noise_sigma)
        self.fz_noise_sigma = float(fz_noise_sigma)

    # ---- Images: brightness only ---- #
    def apply_to_image(self, img_pil: Image.Image) -> torch.Tensor:
        img = TF.adjust_brightness(img_pil, self.brightness)
        tensor = _base_img_transform(img)
        return tensor

    # ---- Position augmentation: Gaussian noise ---- #
    def apply_to_position(self, seq: np.ndarray) -> np.ndarray:
        if self.pos_noise_sigma > 0:
            noise = np.random.randn(len(seq)) * self.pos_noise_sigma
            return seq + noise
        return seq.copy()

    # ---- Force augmentation: Gaussian noise then clamp to <= 0 ---- #
    def apply_to_fz(self, seq: np.ndarray) -> np.ndarray:
        if self.fz_noise_sigma > 0:
            noise = np.random.randn(len(seq)) * self.fz_noise_sigma
            s = seq + noise
        else:
            s = seq.copy()
        s = np.minimum(s, 0.0)
        return s


class TotalDataset(torch.utils.data.Dataset):
    def __init__(self, augment_repeats: int = 1, seed: int = 777,
                 img_brightness_delta: float = 0.15,
                 pos_noise_sigma: float = 0.00005,
                 fz_noise_sigma: float = 0.00005):
        """
        Args:
            augment_repeats: each base sample appears this many times with different augmentation (set 50 for 50x)
            seed: base seed to make per-sample deterministic randomness reproducible
            img_brightness_delta: max absolute brightness change factor around 1.0
            pos_noise_sigma: std for Gaussian noise on x/y sequences
            fz_noise_sigma: std for Gaussian noise on Fz sequence (clamped <= 0 afterward)
        """
        super().__init__()
        assert augment_repeats >= 1
        self.augment_repeats = int(augment_repeats)
        self.base_seed = int(seed)
        self.img_brightness_delta = float(img_brightness_delta)
        self.pos_noise_sigma = float(pos_noise_sigma)
        self.fz_noise_sigma = float(fz_noise_sigma)

        self.demon_num = demon_num
        self.num_arr = np.zeros((self.demon_num, 1), dtype=int)

        # Preload index ranges per demonstration (group)
        for i in range(self.demon_num):
            tmp_images_folder = os.listdir(os.path.join(images_dir, str(i + 1)))
            len_folder = len(tmp_images_folder)
            # number of training windows inside this demo
            self.num_arr[i] = int((len_folder - prediction_horizon) / action_horizon)

        self.group_ranges = np.cumsum(self.num_arr)

        # Accumulate full-series for global min/max (positions, Fz preprocessed)
        self.car_x_total = []
        self.car_y_total = []
        self.Fz_total = []

        for i in range(self.demon_num):
            tmp_csv = pd.read_csv(os.path.join(csv_dir, f"{i+1}.csv"))
            tmp_car_x = np.asarray(tmp_csv['x'], dtype=float)
            tmp_car_y = np.asarray(tmp_csv['y'], dtype=float)
            tmp_Fz = np.asarray(tmp_csv['fz'], dtype=float)

            # Fz preprocessing: low-pass then clamp positive to 0
            tmp_Fz = lowpass_filter(tmp_Fz, wn=0.12, order=2)
            tmp_Fz[tmp_Fz > 0] = 0.0

            self.car_x_total.append(tmp_car_x)
            self.car_y_total.append(tmp_car_y)
            self.Fz_total.append(tmp_Fz)

        # Global min/max for normalization (avoid div-by-zero)
        self.car_x_total = np.concatenate(self.car_x_total)
        # self.min_car_x = float(np.min(self.car_x_total)); self.max_car_x = float(np.max(self.car_x_total))

        self.car_y_total = np.concatenate(self.car_y_total)
        # self.min_car_y = float(np.min(self.car_y_total)); self.max_car_y = float(np.max(self.car_y_total))

        self.Fz_total = np.concatenate(self.Fz_total)
        self.min_Fz = float(np.min(self.Fz_total)); self.max_Fz = float(np.max(self.Fz_total))

        # print(f"===== Car X -> Min: {self.min_car_x}, Max: {self.max_car_x} =====")
        # print(f"===== Car Y -> Min: {self.min_car_y}, Max: {self.max_car_y} =====")
        print(f"===== Fz -> Min: {self.min_Fz}, Max: {self.max_Fz} =====")

        # cache lengths
        self._base_len = int(self.group_ranges[-1])

    def __len__(self):
        return self._base_len * self.augment_repeats

    def _locate_group(self, base_idx: int) -> Tuple[int, int]:
        """Return (group_idx, obs_num) for a base index (no augmentation).
        base_idx in [0, _base_len-1]
        """
        for group_idx, end in enumerate(self.group_ranges):
            if base_idx + 1 <= end:
                if group_idx == 0:
                    obs_num = action_horizon * base_idx
                else:
                    obs_num = action_horizon * (base_idx - int(self.group_ranges[group_idx - 1]))
                return group_idx, int(obs_num)
        raise IndexError("base_idx out of range")

    def _read_images(self, group_idx: int, obs_num: int) -> Tuple[Image.Image, Image.Image]:
        folder = os.path.join(images_dir, str(group_idx + 1))
        tmp_images = sorted(os.listdir(folder), key=lambda x: int(re.findall(r'(\d+)$', os.path.splitext(x)[0])[0]))
        img1 = Image.open(os.path.join(folder, tmp_images[obs_num])).convert('RGB')
        img2 = Image.open(os.path.join(folder, tmp_images[obs_num + 1])).convert('RGB')
        return img1, img2

    def _read_series(self, group_idx: int):
        df = pd.read_csv(os.path.join(csv_dir, f"{group_idx + 1}.csv"))
        car_x = np.asarray(df['x'], dtype=float)
        car_y = np.asarray(df['y'], dtype=float)
        Fz = np.asarray(df['fz'], dtype=float)
        # keep Fz preprocessing consistent with global stats
        Fz = lowpass_filter(Fz, wn=0.12, order=2)
        Fz[Fz > 0] = 0.0
        return car_x, car_y, Fz

    def _normalize(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        return (x - lo) / denom

    def __getitem__(self, idx):
        # Map to (base sample, aug variant)
        base_idx = idx // self.augment_repeats
        aug_id = idx % self.augment_repeats

        group_idx, obs_num = self._locate_group(base_idx)

        # Read raw data
        img1_pil, img2_pil = self._read_images(group_idx, obs_num)
        car_x, car_y, Fz = self._read_series(group_idx)

        # Slice window
        start = obs_num
        end_pred = obs_num + prediction_horizon
        x_win = car_x[start:end_pred]
        y_win = car_y[start:end_pred]
        fz_win = Fz[start:end_pred]

        # Build deterministic RNG per (demo, start, aug_id)
        seed = (self.base_seed * 1_000_003 + (group_idx + 1) * 10007 + (start + 1) * 101 + aug_id) & 0xFFFFFFFF
        rng = random.Random(seed)
        augmenter = Augmenter(rng,
                              img_brightness_delta=self.img_brightness_delta,
                              pos_noise_sigma=self.pos_noise_sigma,
                              fz_noise_sigma=self.fz_noise_sigma)

        # Apply augmentation
        img1 = augmenter.apply_to_image(img1_pil)
        img2 = augmenter.apply_to_image(img2_pil)

        x_aug = augmenter.apply_to_position(x_win)
        y_aug = augmenter.apply_to_position(y_win); y_aug = -y_aug
        fz_aug = augmenter.apply_to_fz(fz_win)

        # Normalize with global stats
        # x_norm = self._normalize(x_aug, self.min_car_x, self.max_car_x)
        # y_norm = self._normalize(y_aug, self.min_car_y, self.max_car_y)
        fz_norm = self._normalize(fz_aug, self.min_Fz, self.max_Fz)

        # Return tensors
        x_t = torch.from_numpy(x_aug.astype(np.float32))
        y_t = torch.from_numpy(y_aug.astype(np.float32))
        fz_t = torch.from_numpy(fz_norm.astype(np.float32))

        return img1, img2, x_t, y_t, fz_t

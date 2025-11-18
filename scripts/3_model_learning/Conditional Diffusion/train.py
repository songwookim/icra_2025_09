import torch
from tqdm import tqdm
from torch.optim import Adam
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from network import *
from HyperParameters import *
from datasets_aug import TotalDataset
# from datasets_org import TotalDataset
from torch.utils.tensorboard import SummaryWriter
from Model import ConditionalUnet1D
from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== Plot Config (Y-axis fixed limits) ===== #
# 고정 y축을 쓰려면 True, 자동 스케일이면 False
FIX_YLIM = True
# 모든 채널에 동일한 범위를 쓰려면 아래 사용
YLIM_ALL = (0, 1.0)
# 채널별로 다른 범위를 쓰려면 리스트로 지정 (예: x, y, Fz)
YLIMS_PER_CHANNEL = [(0, 0.5), (0, 0.5), (0, 1)] 

# ===== Vision Encoder ===== #
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()

os.makedirs(logs_base_dir, exist_ok=True)
writer = SummaryWriter(logs_base_dir)

# ===== Download Datasets(Customized Dataset) ===== #
transform = transforms.Compose([transforms.ToTensor(),])

trajectory_dataset = TotalDataset(
    augment_repeats=50,
    img_brightness_delta=0.15,
    pos_noise_sigma=0.0005,
    fz_noise_sigma=0.0005
)
dataset_size = len(trajectory_dataset)
train_num = dataset_size - test_num

train_dataset, test_dataset = random_split(trajectory_dataset, [train_num, test_num])
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=True)

noise_pred_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=observation_horzion*obs_dim)
# ===== Diffusion Model Training ===== #

diffusion = Diffusion(noise_pred_net, n_times=n_timesteps, beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)

optimizer = Adam(diffusion.parameters(), lr=lr)
denoising_loss = nn.MSELoss()

print("Start training DDPMs...")
noise_pred_net.train()

for epoch in range(epochs):
    noise_prediction_loss = 0
    for batch_idx, x in enumerate(train_loader):
        optimizer.zero_grad()

        # ----- Observation Preprocessing ----- #
        img1 = x[0].to(DEVICE).float()
        img2 = x[1].to(DEVICE).float()
        car_x = x[2].to(DEVICE).float()
        car_y = x[3].to(DEVICE).float()
        Fz = x[4].to(DEVICE).float()

        # Image Visual Encoder
        with torch.no_grad():
            img1 = resnet(img1).unsqueeze(1)
            img2 = resnet(img2).unsqueeze(1)

        img_concat = torch.concat((img1, img2), dim=1) # img_concat shape: (B, TimeLength, Image Dimension)

        pose_concat = torch.concat((car_x[:, :2].unsqueeze(2), car_y[:, :2].unsqueeze(2)), dim=2)   # pose_concat shape: (B, TimeLength, Position Dimension)
        condition = torch.concat((img_concat, pose_concat), dim=2)
        condition = condition.reshape(condition.shape[0], -1)   # obs shape:(B, Observation Dimension)

        # ----- Action Preprocessing ----- #
        car_concat = torch.concat((car_x.unsqueeze(2), car_y.unsqueeze(2)), dim=2)
        x = torch.concat((car_concat, Fz.unsqueeze(2)), dim=2)  # (B, Time Length, Action Dimension) 예상

        noisy_input, epsilon, pred_epsilon = diffusion(x, condition)
        loss = denoising_loss(pred_epsilon, epsilon)

        noise_prediction_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss: ", noise_prediction_loss / batch_idx)
    writer.add_scalar('Loss/train', noise_prediction_loss / batch_idx, epoch)

print("===== Train Finished!! =====")
os.makedirs("./check_point", exist_ok=True)
torch.save(noise_pred_net.state_dict(), f"./check_point/diffusion_model_v%d_%d.pt" % (exp_ver, exp_num))

# ===== Load and Rollout ===== #
noise_pred_net.load_state_dict(torch.load(ckpt_dir))
diffusion = Diffusion(noise_pred_net, n_times=n_timesteps, beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)

noise_pred_net.eval()
os.makedirs('plots', exist_ok=True)

for batch_idx, x in enumerate(test_loader):
    img1 = x[0].to(DEVICE).float()
    img2 = x[1].to(DEVICE).float()
    car_x = x[2].to(DEVICE).float()
    car_y = x[3].to(DEVICE).float()
    Fz = x[4].to(DEVICE).float()

    # Image Visual Encoder
    with torch.no_grad():
        img1 = resnet(img1).unsqueeze(1)
        img2 = resnet(img2).unsqueeze(1)

    img_concat = torch.concat((img1, img2), dim=1)

    pose_concat = torch.concat((car_x[:, :2].unsqueeze(2), car_y[:, :2].unsqueeze(2)), dim=2)
    condition = torch.concat((img_concat, pose_concat), dim=2)
    condition = condition.reshape(condition.shape[0], -1)

    car_concat = torch.concat((car_x.unsqueeze(2), car_y.unsqueeze(2)), dim=2)
    x = torch.concat((car_concat, Fz.unsqueeze(2)), dim=2)  # (B, 16, 3) 예상

    with torch.no_grad():
        generated_profile = diffusion.sample(N=inference_batch_size, cond=condition)  # (B, 16, 3) 예상

    # ====== Plot: act vs generated_profile (첫 번째 샘플만) ======
    try:
        # act: torch -> numpy
        x_np = x[0].detach().cpu().numpy()          # (16, 3)
        # generated_profile: torch -> numpy
        if isinstance(generated_profile, torch.Tensor):
            gen_np = generated_profile[0].detach().cpu().numpy()  # (16, 3)
        else:
            gen_np = np.array(generated_profile)
            if gen_np.ndim == 3 and gen_np.shape[0] == 1:
                gen_np = gen_np[0]

        # 시간축
        T = x_np.shape[0]
        t = np.arange(T)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        channel_names = ['Positiion X', 'Position Y', 'Force Z']  # 필요시 ['x', 'y', 'Fz']로 변경

        for i in range(3):
            axes[i].plot(t, x_np[:, i], label='act')
            axes[i].plot(t, gen_np[:, i], label='generated_profile')
            axes[i].set_ylabel(channel_names[i])
            axes[i].grid(True, linestyle='--', alpha=0.4)

            # === Y-axis 고정 범위 적용 ===
            if FIX_YLIM:
                if YLIMS_PER_CHANNEL is not None:
                    lo, hi = YLIMS_PER_CHANNEL[i]
                else:
                    lo, hi = YLIM_ALL
                axes[i].set_ylim(lo, hi)

        axes[-1].set_xlabel('timestep')
        # ---- Legend: 상단 오른쪽으로 고정 ----
        axes[0].legend(loc='upper right', frameon=True)

        out_path = f'plots/rollout_compare_batch{batch_idx}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved plot] {out_path}")
    except Exception as e:
        print(f"[Plot skipped] {e}")

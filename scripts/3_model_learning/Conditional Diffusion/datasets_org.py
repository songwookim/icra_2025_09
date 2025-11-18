import os
import torch
import pandas as pd
import numpy as np
from config import *
from PIL import Image
from torchvision import transforms
import re
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt

transform = transforms.Compose([
    transforms.ToTensor(),  # (H, W, C) -> (C, H, W), 0~255 -> 0~1
    transforms.Resize((128, 128)),  # 크기 조정
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

def lowpass_filter(data, cutoff=2.0, fs=8.9, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

scaler = MinMaxScaler()

class TotalDataset(torch.utils.data.Dataset):
    def __init__(self):
        
        self.demon_num = demon_num
        self.num_arr = np.zeros((self.demon_num, 1), dtype=int)
        self.car_x_total = []; self.car_y_total = []; self.Fz_total = []

        for i in range(self.demon_num):
            tmp_images_folder = os.listdir(images_dir + '/' + str(i + 1))
            len_folder = len(tmp_images_folder)

            count = 0
            for start in range(0, len_folder - prediction_horizon, action_horizon):
                count += 1

            self.num_arr[i] = count

            tmp_csv = pd.read_csv(csv_dir + '/' + str(i+1) + '.csv')
            tmp_car_x = tmp_csv['x']; tmp_car_x = np.array(tmp_car_x, dtype=float); self.car_x_total.append(tmp_car_x)
            tmp_car_y = tmp_csv['y']; tmp_car_y = np.array(tmp_car_y, dtype=float); self.car_y_total.append(tmp_car_y)
            tmp_Fz = tmp_csv['fz']; tmp_Fz = np.array(tmp_Fz, dtype=float); 
            
            # ----- Fz Preprocessing ----- #
            tmp_Fz = lowpass_filter(tmp_Fz)  
            tmp_Fz[tmp_Fz > 0] = 0  
            self.Fz_total.append(tmp_Fz)
            
        self.group_ranges = np.cumsum(self.num_arr)

        # ----- CSV Normalizaiton ----- #
        self.car_x_total = np.concatenate(self.car_x_total)
        self.car_y_total = np.concatenate(self.car_y_total)
        self.Fz_total = np.concatenate(self.Fz_total); self.min_Fz = np.min(self.Fz_total); self.max_Fz = np.max(self.Fz_total)

        print(f"===== Fz -> Min: %s, Max: %s =====" %(self.min_Fz, self.max_Fz))
        
        
    def __len__(self):
        return self.group_ranges[-1]

    def __getitem__(self, idx):
        for group_idx, end in enumerate(self.group_ranges):
            if idx+1 <=end:
                if group_idx == 0:
                    obs_num = action_horizon*idx
                    # ----- Observation Images ----- #
                    tmp_images = os.listdir(images_dir + '/' + str(group_idx+1))
                    tmp_images = sorted(
                        tmp_images,
                        key=lambda x: int(re.findall(r'(\d+)$', os.path.splitext(x)[0])[0]),
                        reverse=False
                    )

                    image1 = Image.open(images_dir + '/' + str(group_idx+1) + '/' + tmp_images[obs_num])
                    image1 = np.array(image1)
                    image1 = transform(image1)

                    image2 = Image.open(images_dir + '/' + str(group_idx+1) + '/' + tmp_images[obs_num+1])
                    image2 = np.array(image2)
                    image2 = transform(image2)

                    # ----- Position Force ----- #
                    pos_force = pd.read_csv(csv_dir + '/' + str(group_idx+1) + '.csv')
                    car_x = pos_force['x']; car_x = np.array(car_x, dtype=float)
                    car_y = pos_force['y']; car_y = np.array(car_y, dtype=float)
                    Fz = pos_force['fz']; Fz = np.array(Fz, dtype=float); Fz = (Fz - self.min_Fz) / (self.max_Fz - self.min_Fz + 1e8)

                    return image1, image2, car_x[obs_num:obs_num+prediction_horizon], car_y[obs_num:obs_num+prediction_horizon], Fz[obs_num:obs_num+prediction_horizon]

                else:
                    obs_num = action_horizon*(idx - self.group_ranges[group_idx-1])
                    # ----- Observation Images ----- #
                    tmp_images = os.listdir(images_dir + '/' + str(group_idx+1))
                    tmp_images = sorted(
                    tmp_images,
                    key=lambda x: int(re.findall(r'(\d+)$', os.path.splitext(x)[0])[0]),
                    reverse=False
                    )
                    
                    image1 = Image.open(images_dir + '/' + str(group_idx+1) + '/' + tmp_images[obs_num])
                    image1 = np.array(image1)
                    image1 = transform(image1)

                    image2 = Image.open(images_dir + '/' + str(group_idx+1) + '/' + tmp_images[obs_num+1])
                    image2 = np.array(image2)
                    image2 = transform(image2)

                    # ----- Position Force ----- #
                    pos_force = pd.read_csv(csv_dir + '/' + str(group_idx+1) + '.csv')
                    car_x = pos_force['x']; car_x = np.array(car_x, dtype=float)
                    car_y = pos_force['y']; car_y = np.array(car_y, dtype=float)
                    Fz = pos_force['fz']; Fz = np.array(Fz, dtype=float); Fz = (Fz - self.min_Fz) / (self.max_Fz - self.min_Fz)

                    return image1, image2, car_x[obs_num:obs_num+prediction_horizon], car_y[obs_num:obs_num+prediction_horizon], Fz[obs_num:obs_num+prediction_horizon]


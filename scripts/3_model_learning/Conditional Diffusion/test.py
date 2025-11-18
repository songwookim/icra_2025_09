import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from network import *
from HyperParameters import *
from datasets import TrajectoryTestDataset
from config import *
import os


def draw_sample_image(x, postfix, idx):
    plt.figure(idx, figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))


def denormalize(data, min_value, max_value):
    normalized_range = max_value - min_value
    denormalized_data = (data * normalized_range) + min_value
    return denormalized_data


model_path = ckpt_dir

if not os.path.exists(model_path):
    exit()
# ------------------------------------------------------
# Download Datasets
transform = transforms.Compose([transforms.ToTensor(), ])

kwargs = {'num_workers': 0, 'pin_memory': True}

trajectory_dataset_test = TrajectoryTestDataset()

test_loader = DataLoader(dataset=trajectory_dataset_test, batch_size=inference_batch_size, shuffle=False)

# ------------------------------------------------------
noise_pred_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=observation_horzion*obs_dim)()

model.load_state_dict(torch.load(model_path))

diffusion = Diffusion(model, n_times=n_timesteps, beta_minmax=beta_minmax, device=DEVICE).to(
    DEVICE)

# Visualizing result
model.eval()
for batch_idx, x in enumerate(test_loader):
    origin_image = x
    x = x.to(DEVICE).float()
    perturbed_images, epsilon, pred_epsilon = diffusion(x)
    break

with torch.no_grad():
    generated_images = diffusion.sample(N=inference_batch_size)
    generated_images = generated_images.cpu()
    generated_images = np.reshape(generated_images, [60, 64*4])
    df_generated = pd.DataFrame(generated_images)

df_generated.to_csv('./generated_data_d6_v12.csv', header=False, index=False)

print("Done!")

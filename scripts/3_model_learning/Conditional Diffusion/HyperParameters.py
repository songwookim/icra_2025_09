import torch
import numpy as np

# ------------------------------------------------------
# Hyper Parameters
cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

timestep_embedding_dim = 256
n_timesteps = 1000
beta_minmax = [1e-4, 2e-2]

n_layers = 8
hidden_dim = 256
hidden_dims = [hidden_dim for _ in range(n_layers)]

inference_batch_size = 1

seed = 1234
torch.manual_seed(1234)
np.random.seed(seed)
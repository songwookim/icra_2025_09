# Experimental Settings
exp_ver = 1
exp_num = 1

# Hyperparameters
demon_num = 10

prediction_horizon = 24
action_horizon = 20
observation_horzion = 2

vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 3

# Train & Test Hyperparameter
test_num = 10
train_batch_size = 4
lr = 0.0001
epochs = 500

# Directory
images_dir = "./datasets/images"
csv_dir = "./datasets/positon_force"
logs_base_dir = f"./logs/%d_%d/" %(exp_ver, exp_num)
ckpt_dir = './check_point/diffusion_model_v1_1.pt'

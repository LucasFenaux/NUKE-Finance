import torch


game = 'trading'
gamma = 0.99
batch_size = 128
fe_lr = 1e-5
am_lr = 1e-5
initial_exploration = 128
replay_capacity = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epsilon = 0.05
epochs = 10000
update_target = 100
train_frequency = int(batch_size / 8)  # we train as much as we would if we trained every epoch with batch size 8
sequence_length = 100
num_actions = 5
model_num = 0
starting_liquidity = 1000000
save_dir = "/scratch/lprfenau/datasets/nuke/"
week_config = (20, 0.1, 100, 100, 5)
month_config = (20, 0.1, 100, 1000, 5)
year_config = (20, 0.1, 100, 80, 5)
use_cached_pre_processing = True
use_cached_pre_computing = True
log_frequency = 100
alpha = 1e-5

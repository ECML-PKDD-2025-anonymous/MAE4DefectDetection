import os
import torch.nn as nn
from piqa import SSIM, PSNR

# Construct the path to the SAM_Images_cropped directory 
current_dir = os.getcwd()
sam_crops_directory = os.path.abspath(os.path.join(current_dir, 'Data', 'SAM_Images_cropped', 'all_cycles_prediction'))
mask_directory = os.path.abspath(os.path.join(current_dir, 'Data', 'SAM_segmentation_masks'))
tta_pkl_path = os.path.abspath(os.path.join(current_dir, 'Data', 'TTA_Dataframes', 'RUL_dataframe.pkl'))
train_rul_path = os.path.abspath(os.path.join(current_dir, 'Data', 'TTA_Dataframes', 'train_RUL_dataframe.pkl'))
test_rul_path = os.path.abspath(os.path.join(current_dir, 'Data', 'TTA_Dataframes', 'test_RUL_dataframe.pkl'))
eval_rul_path = os.path.abspath(os.path.join(current_dir, 'Data', 'TTA_Dataframes', 'eval_RUL_dataframe.pkl'))
image_net_path = os.path.abspath(os.path.join(current_dir, 'Data'))
moving_mnist_directory = os.path.abspath(os.path.join(current_dir, 'Data'))

# Directories where models and performance measures for tensorboard are saved
model_dir = os.path.abspath(os.path.join(current_dir, 'Training', 'Trained_Models'))
log_dir = os.path.abspath(os.path.join(current_dir, 'Training', 'Training_Logs'))

# SAM preprocessing constants 
timestamps = ["0000TSC", "0100TSC", "0500TSC", "1000TSC", "1500TSC"]
panel_types = ["INDIUM", "INNOLOT", "SAC105", "SAC305", "SENJU"]
led_types = ['cree', 'dominant', 'lumileds', 'nichia131', 'nichia170', 'osrambf', 'osramcomp', 'samsung', 'seoul']
led_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# Parameters for creating Trainer Object 
LOSS_DICT = {"mse": nn.MSELoss(), "ssim": SSIM(n_channels=1, value_range=1.0), "psnr": PSNR(value_range=1.0)}
NORMALIZATION_DICT = {"sam": [0.2221, 0.1491]} 
VIT_DICT = {
    "ti": {"depth": 12, "embed_dim": 192, "heads": 3},
    "s": {"depth": 12, "embed_dim": 384, "heads": 6},
    "b": {"depth": 12, "embed_dim": 768, "heads": 12},
    "l": {"depth": 24, "embed_dim": 1024, "heads": 16}
}
# src/config.py
# Handle Library Imports
import os
import sys
import torch
import torch.nn as nn

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Handle Module imports
from src.models.unet_models import UNet, ResUNet, AttentionUNet, ResAttentionUNet
from src.models.losses import (
    DiceWithLogitsLoss, FocalWithLogitsLoss, DiceBCEWithLogitsLoss,
    DiceFocalWithLogitsLoss, DiceFocalBCEWithLogitsLoss,
    GradLoss, SSIMLoss, GradSSIMLoss, MSEGradLoss, MSESSIMLoss, MSESSIMGradLoss
)

config = {
    'data_path': '../data', # Path to data directory (containing all datasets which have 3directories (rgb, sar, dsm))
    'dataset_name' : "ieeegrss_2023dfc_track2",
    'dataset_type': 'regression',
    'normalization' : 1,
    'split' : 0.9,
    'task_type' : 'regression',
    'composition' : 0,
    'composition_type': None,
    'base_net': UNet,
    'segmentation_net_path': None,  # Path to the pretrained segmentation_net
    'in_channels': 4,
    'out_channels': 1,
    'dropout' : 0,
    'batchnorm' : True,
    'bias' : False,
    'pooling' : 'spectralpool',
    'batch_size': 4,
    'learning_rate': 0.001,
    'epochs': 80,
    'device':  'cuda' if torch.cuda.is_available() else 'cpu',
    'loss_fn': nn.MSELoss(),
    'save_path' : None,
    'save_name' : None
}

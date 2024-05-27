# src/experiments/experiment_06/E06_config.py
# Handle Library imports
import os
import sys
import torch

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Handle Module Imports
from src.models.unet_models import UNet, ResUNet, AttentionUNet, ResAttentionUNet

segmentation_net_config = {
    'base_net': ResAttentionUNet,
    'in_channels': 4,
    'out_channels': 1,
    'dropout' : 0,
    'batchnorm' : True,
    'bias' : False,
    'pooling' : 'spectralpool'
}

e06_config = {
    'data_path': '../../../data', # Path to data directory (containing all datasets which have 3directories (rgb, sar, dsm))
    'dataset_name' : "ieeegrss_2023dfc_track2",
    'dataset_type': 'regression',
    'normalization' : 1,
    'split' : 0.9,
    'task_type' : 'regression',
    'composition' : 1,
    'composition_type': 2,
    'base_net': ResAttentionUNet,
    'segmentation_net_path': '../../../models_archive/segmentation_models/segmentation_resattentionunet.pth',  # Path to the pretrained segmentation_net
    'segmentation_net_config' : segmentation_net_config,
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
    'loss_fn': None,
    'save_path' : None,
    'save_name' : None
}
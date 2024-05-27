# src/experiments/experiment_02/E02_config.py
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

e02_config = {
    'data_path': '../../../data', # Path to data directory (containing all datasets which have 3directories (rgb, sar, dsm))
    'dataset_name' : "ieeegrss_2023dfc_track2",
    'dataset_type': 'segmentation',
    'normalization' : 1,
    'split' : 0.9,
    'task_type' : 'segmentation',
    'composition' : 0,
    'composition_type': None,
    'base_net': ResAttentionUNet,
    'segmentation_net_path': None,  # Path to the pretrained segmentation_net
    'segmentation_net_config' : None,
    'in_channels': 4,
    'out_channels': 1,
    'dropout' : 0,
    'batchnorm' : True,
    'bias' : False,
    'pooling' : 'spectralpool',
    'batch_size': 80,
    'learning_rate': 0.001,
    'epochs': 1,
    'device':  'cuda' if torch.cuda.is_available() else 'cpu',
    'loss_fn': None,
    'save_path' : None,
    'save_name' : None
}
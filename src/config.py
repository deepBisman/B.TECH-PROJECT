import torch
import torch.nn as nn
from src.models.unet_models import UNet
# src/config.py
config = {
    'data_path': '../data', # Path to data directory (containing all datasets which have 3directories (rgb, sar, dsm))
    'dataset_name' : "ieeegrss_2023dfc_track2",
    'dataset_type': 'regression',
    'normalization' : 1,
    'split' : 0.9,
    'task_type' : 'regression',
    'composition' : 0,
    'base_net': UNet,
    'segmentation_net_path': '../pretrained_models/unet1.pth',  # Path to the pretrained segmentation_net
    'composition_type': None,
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

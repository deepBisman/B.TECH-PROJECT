# Handle Library Imports
import os
import sys
import random
import torch
import torch.optim as optim
import torch.nn as nn

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Handle module imports
from src.config import config
from src.data.dataset_utils import visualize_dataset, dataset_init
from src.models.unet_models import UNet, ResUNet, AttentionUNet, ResAttentionUNet
from src.models.composite_models import TwoStageUNetSequential, TwoStageUNetFusion
from src.models.losses import (
    DiceWithLogitsLoss, FocalWithLogitsLoss, DiceBCEWithLogitsLoss,
    DiceFocalWithLogitsLoss, DiceFocalBCEWithLogitsLoss,
    GradLoss, SSIMLoss, GradSSIMLoss, MSEGradLoss, MSESSIMLoss, MSESSIMGradLoss
)
from src.utils.metrics import segmentation_metrics, regression_metrics
from src.utils.utils import (
    train, test, hyperparameter_tuning, model_init, 
    visualize_model_output, load_pretrained_net, model_evaluator
)

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Desired Device
    device = config['device']
    
    # Initialize dataset dictionery
    data_dict = dataset_init(config, seed)
    
    # Assign dataset and dataloader objects
    dataset, train_dataset, test_dataset = data_dict['dataset'], data_dict['train_dataset'], data_dict['test_dataset']
    train_loader, test_loader = data_dict['train_loader'], data_dict['test_loader']
    
    # Visualize Dataset
    print("----------------Visualizing Dataset-----------------")
    visualize_dataset(dataset, config['dataset_type'])

    # Initialize the model
    model = model_init(config)

    # Define loss function and optimizer
    criterion = config['loss_fn']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    
    # Train the model
    train(model, train_loader, test_loader, optimizer, criterion, device, config['epochs'], config['save_name'], config['save_path'])
        
    # Visualize model outputs on train and test data
    visualize_model_output(model, train_dataset, device, message="------------------TRAINING SET MODEL OUTPUTS-------------------")
    visualize_model_output(model, test_dataset, device, message="------------------TESTING SET MODEL OUTPUTS-------------------")

    # Evaluate the Model on Train and Test Sets and print the corresponding metric
    _, _ = model_evaluator(model, train_loader, test_loader, device, config)

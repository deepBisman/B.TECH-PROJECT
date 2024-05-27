# Handle Library Imports
import os
import sys
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from tabulate import tabulate

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Handle Module Imports
from src.data.dataset_utils import dataset_init, visualize_dataset
from src.models.unet_models import UNet, ResUNet, AttentionUNet, ResAttentionUNet
from src.utils.utils import suppress_output, model_evaluator, display_experiment_results
from src.utils.utils import (
    train, model_init, model_evaluator,
    suppress_output, display_experiment_results
)
from src.experiments.experiment_01.E01_config import e01_config


if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # Desired Device
    device = e01_config['device']

    # Define set of models for testing
    BASE_NET = [UNet, ResUNet, AttentionUNet, ResAttentionUNet]

    # Initialize dataset dictionary
    data_dict = dataset_init(e01_config, seed)

    # Assign dataset and dataloader objects
    dataset, train_dataset, test_dataset = data_dict['dataset'], data_dict['train_dataset'], data_dict['test_dataset']
    train_loader, test_loader = data_dict['train_loader'], data_dict['test_loader']

    # Visualize Dataset
    print("----------------Visualizing Dataset-----------------")
    visualize_dataset(dataset, e01_config['dataset_type'])

    # Store Model Evaluations in a Dictionary
    evaluations = {}

    # Iterate over models with progress bar
    for net in tqdm(BASE_NET, desc="Training and Evaluating Models", unit="model"):
        # Set the config parameter 'base_net'
        e01_config['base_net'] = net

        # Initialize the model
        model = model_init(e01_config)

        # Define loss function and optimizer
        criterion = e01_config['loss_fn']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=e01_config['learning_rate'])

        # Train the model
        with suppress_output():
            train(model, train_loader, test_loader, optimizer, criterion, device, e01_config['epochs'], e01_config['save_name'], e01_config['save_path'])

        # Evaluate the Model on Train and Test Sets and print the corresponding metric
        with suppress_output():
            evaluations[f'{net.__name__}'] = model_evaluator(model, train_loader, test_loader, device, e01_config)

    # Display Experiment Results
    print("---------------------Training Metrics:------------------------")
    print(display_experiment_results(evaluations, 0, "Model"))
    print("---------------------Testing  Metrics:------------------------")
    print(display_experiment_results(evaluations, 1, "Model"))

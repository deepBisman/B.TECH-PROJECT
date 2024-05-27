# Handle Library imports
import os
import sys
import torch
import torch.optim as optim
import random
from tqdm import tqdm

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Handle Module Imports
from src.data.dataset_utils import dataset_init, visualize_dataset
from src.models.losses import (
    DiceWithLogitsLoss, FocalWithLogitsLoss, DiceBCEWithLogitsLoss,
    DiceFocalWithLogitsLoss, DiceFocalBCEWithLogitsLoss
)
from src.utils.utils import (
    train, model_init, model_evaluator,
    suppress_output, display_experiment_results
)
from src.experiments.experiment_02.E02_config import e02_config

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # Desired Device
    device = e02_config['device']

    # Define set of loss functions for testing
    LOSS_FN = {
        'BCE Loss': torch.nn.BCEWithLogitsLoss(), 
        'Dice Loss': DiceWithLogitsLoss(), 
        'Focal Loss': FocalWithLogitsLoss(alpha=torch.tensor([0.25, 0.75])), 
        'Dice + BCE Loss': DiceBCEWithLogitsLoss(),
        'Dice + Focal Loss': DiceFocalWithLogitsLoss(alpha=torch.tensor([0.25, 0.75])), 
        'Dice + BCE + Focal Loss': DiceFocalBCEWithLogitsLoss(alpha=torch.tensor([0.25, 0.75]))
    }

    # Initialize dataset dictionary
    data_dict = dataset_init(e02_config, seed)

    # Assign dataset and dataloader objects
    dataset, train_dataset, test_dataset = data_dict['dataset'], data_dict['train_dataset'], data_dict['test_dataset']
    train_loader, test_loader = data_dict['train_loader'], data_dict['test_loader']

    # Visualize Dataset
    print("----------------Visualizing Dataset-----------------")
    visualize_dataset(dataset, e02_config['dataset_type'])

    # Store Model Evaluations in a Dictionary
    evaluations = {}

    # Iterate over loss functions with progress bar
    for loss_name, loss_fn in tqdm(LOSS_FN.items(), desc="Training and Evaluating Models", unit="loss_fn"):
        # Set the config parameter 'loss_fn' to a new instance of the loss function
        e02_config['loss_fn'] = loss_fn

        # Initialize the model
        model = model_init(e02_config)

        # Define loss function and optimizer
        criterion = e02_config['loss_fn']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=e02_config['learning_rate'])

        # Train the model
        with suppress_output():
            train(model, train_loader, test_loader, optimizer, e02_config['loss_fn'], device, e02_config['epochs'], e02_config['save_name'], e02_config['save_path'])

        # Evaluate the Model on Train and Test Sets and print the corresponding metric
        with suppress_output():
            evaluations[loss_name] = model_evaluator(model, train_loader, test_loader, device, e02_config)

    # Display Experiment Results
    print("---------------------Training Metrics:------------------------")
    print(display_experiment_results(evaluations, 0, "Loss Function"))
    print("---------------------Testing  Metrics:------------------------")
    print(display_experiment_results(evaluations, 1, "Loss Function"))

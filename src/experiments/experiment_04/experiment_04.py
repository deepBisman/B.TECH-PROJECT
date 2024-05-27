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

# Handle Module imports
from src.data.dataset_utils import dataset_init, visualize_dataset
from src.utils.utils import (
    train, model_init, model_evaluator,
    suppress_output, display_experiment_results
)
from src.experiments.experiment_04.E04_config import e04_config

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # Desired Device
    device = e04_config['device']

    # Define composition parameteres for testing
    COMPOSITION = {
        # Type of Regression Model : (e04_config[composition], e04_config[composition_type])
        'SSR': (0, None), 
        'TSR_M1' : (1, 1),
        'TSR_M2' : (1, 2)
    }

    # Initialize dataset dictionary
    data_dict = dataset_init(e04_config, seed)

    # Assign dataset and dataloader objects
    dataset, train_dataset, test_dataset = data_dict['dataset'], data_dict['train_dataset'], data_dict['test_dataset']
    train_loader, test_loader = data_dict['train_loader'], data_dict['test_loader']

    # Visualize Dataset
    print("----------------Visualizing Dataset-----------------")
    visualize_dataset(dataset, e04_config['dataset_type'])

    # Store the evaluation of models
    evaluations = {}

    # Iterate over model compositions with progress bar
    for model_name, composition_config in tqdm(COMPOSITION.items(), desc="Training and Evaluating Models", unit="Model"):
        # Set the e04_config parameters for composition
        e04_config['composition'], e04_config['composition_type'] = composition_config

        # Initialize the model
        model = model_init(e04_config)

        # Define loss function and optimizer
        criterion = e04_config['loss_fn']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=e04_config['learning_rate'])

        # Train the model on full dataset
        with suppress_output():
            train(model, train_loader, test_loader, optimizer, e04_config['loss_fn'], device, e04_config['epochs'], e04_config['save_name'], e04_config['save_path'])
        
        # Evaluate the Model on Train and Test Sets and print the corresponding metric
        with suppress_output():
            evaluations[model_name] = model_evaluator(model, train_loader, test_loader, device, e04_config)

    # Display Experiment Results
    print("---------------------Training Set Evaluations:------------------------")
    print(display_experiment_results(evaluations, 0, "Model Types"))
    print("---------------------Testing  Set Evaluations:------------------------")
    print(display_experiment_results(evaluations, 1, "Model Types"))

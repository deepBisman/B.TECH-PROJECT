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
from src.data.dataset_utils import dataset_init
from src.utils.utils import (
    train, model_init, model_evaluator,
    suppress_output, display_experiment_results
)
from src.experiments.experiment_07.E07_config import e07_config

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    # Desired Device
    device = e07_config['device']

    # Define set of datasets for testing
    DATASETS = {
        'Berlin': 'berlin', 
        'New York, Sydney & San Diego ' : 'newyork_sandiego_sydney', 
        'Barcelona, Copenhagen & Portsmouth': 'barcelona_copenhagen_portsmouth', 
        'New Delhi, Sao Luis, Brasilia, Rio': 'newdelhi_saoluis_brasilia_rio'
    }

    ################## Train on full dataset, test on subsets#####################################
    # Set the e07_config['dataset_name"] parameter to complete dataset
    e07_config['dataset_name'] = 'ieeegrss_2023dfc_track2'

    # Initialize dataset dictionary
    data_dict = dataset_init(e07_config, seed)

    # Assign dataset and dataloader objects
    dataset, train_dataset, test_dataset = data_dict['dataset'], data_dict['train_dataset'], data_dict['test_dataset']
    train_loader, test_loader = data_dict['train_loader'], data_dict['test_loader']

    # Initialize the model
    model = model_init(e07_config)

    # Define optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=e07_config['learning_rate'])

    # Train the model on full dataset
    train(model, train_loader, test_loader, optimizer, e07_config['loss_fn'], device, e07_config['epochs'], e07_config['save_name'], e07_config['save_path'])

    # Store Model Evaluations in a Dictionary
    cross_evaluations = {}

    # Iterate over sub datasets with progress bar
    for dataset_name, dataset_folder_name in tqdm(DATASETS.items(), desc="Training and Evaluating Models", unit="sub-dataset"):
        # Set the e07_config parameter "dataset-_name"
        e07_config['dataset_name'] = dataset_folder_name

        # Initialize subdataset and their loaders
        sub_data_dict = dataset_init(e07_config, seed)

        # Assign subdataset and subdataloader objects
        sub_dataset, sub_train_dataset, sub_test_dataset = sub_data_dict['dataset'], sub_data_dict['train_dataset'], sub_data_dict['test_dataset']
        sub_train_loader, sub_test_loader = sub_data_dict['train_loader'], sub_data_dict['test_loader']
        
        # Evaluate the Model on Train and Test Sets and print the corresponding metric
        with suppress_output():
            cross_evaluations[dataset_name] = model_evaluator(model, sub_train_loader, sub_test_loader, device, e07_config)

    # Display Experiment Results
    print("---------------------Cross Evaluations:------------------------")
    print(display_experiment_results(cross_evaluations, 1, "City Groups"))
    ################## Train on full dataset, test on subsets#####################################
    
    ################## Train and test on subsets#####################################
    # Store Model Evaluations in a Dictionary
    individual_evaluations = {}

    # Change number of epochs
    e07_config['epochs'] = 25

    # Iterate over sub datasets with progress bar
    for dataset_name, dataset_folder_name in tqdm(DATASETS.items(), desc="Training and Evaluating Models", unit="sub-dataset"):
        # Set the e07_config parameter "dataset-_name"
        e07_config['dataset_name'] = dataset_folder_name

        # Initialize subdataset and their loaders
        sub_data_dict = dataset_init(e07_config, seed)

        # Assign subdataset and subdataloader objects
        sub_dataset, sub_train_dataset, sub_test_dataset = sub_data_dict['dataset'], sub_data_dict['train_dataset'], sub_data_dict['test_dataset']
        sub_train_loader, sub_test_loader = sub_data_dict['train_loader'], sub_data_dict['test_loader']
    
        # Initialize the model
        model = model_init(e07_config)

        # Define loss function and optimizer
        criterion = e07_config['loss_fn']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=e07_config['learning_rate'])

        # Train the model on full dataset
        with suppress_output():
            train(model, sub_train_loader, sub_test_loader, optimizer, e07_config['loss_fn'], device, e07_config['epochs'], e07_config['save_name'], e07_config['save_path'])
        
        # Evaluate the Model on Train and Test Sets and print the corresponding metric
        with suppress_output():
            individual_evaluations[dataset_name] = model_evaluator(model, sub_train_loader, sub_test_loader, device, e07_config)

    # Display Experiment Results
    print("---------------------Individual Evaluations:------------------------")
    print(display_experiment_results(individual_evaluations, 1, "City Groups"))
    ################## Train and test on subsets#####################################

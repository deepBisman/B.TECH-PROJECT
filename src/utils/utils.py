# Handle Library Imports
import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import optuna
from contextlib import contextmanager
from tabulate import tabulate
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Handle Module Imports
from src.models.composite_models import TwoStageUNetFusion, TwoStageUNetSequential
from src.utils.metrics import regression_metrics, segmentation_metrics

# Suppress function outputs
@contextmanager
def suppress_output():
    """Suppress terminal output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Load a pretained model with suppressed function outputs
def load_pretrained_net(path, net_config):
    """
    Load a pre-trained model from a given path with suppressed function outputs.

    Args:
        path (str): The path to the pre-trained model.
        net_config (dict): A dictionary containing configuration parameters for the network.

    Returns:
        torch.nn.Module: The pre-trained model.
    """
    with suppress_output():
        net = net_config['base_net'](in_channels=net_config['in_channels'], out_channels=net_config['out_channels'], dropout=net_config['dropout'], batchnorm=net_config['batchnorm'], bias=net_config['bias'], pooling=net_config['pooling']).to(net_config['device'])
        net.load_state_dict(torch.load(path))
        net.eval()  # Ensure net is in evaluation mode
    return net

# Model Initializer
def model_init(config):
    """
    Initializes the model parameters

    Args:
        config : dictionery containing configuration setting
    """
    # Desired Device
    device = config['device']
    # Check if composition is ON or OFF
    if config['composition'] == 1 :
        # Load pretrained segmentation_net
        segmentation_net = load_pretrained_net(config['segmentation_net_path'], config['in_channels'], config['out_channels'], device)
        # Choose composition type
        if config['composition_type'] == 1:
            # Initialize Regreesion Net
            regression_net = config['base_net'](in_channels=config['in_channels'] + 1, out_channels=config['out_channels'], dropout=config['dropout'], batchnorm=config['batchnorm'], bias=config['bias'], pooling=config['pooling']).to(device)
            # Initialize Composite Model
            model = TwoStageUNetSequential(segmentation_net, regression_net).to(device)
        elif config['composition_type'] == 2:
            # Initialize Regreesion Net
            regression_net = config['base_net'](in_channels=config['in_channels'], out_channels=config['out_channels'], dropout=config['dropout'], batchnorm=config['batchnorm'], bias=config['bias'], pooling=config['pooling']).to(device)
            # Initialize Composite Model
            model = TwoStageUNetFusion(segmentation_net, regression_net).to(device)
    else :
        # Initialize Model Network
        model = config['base_net'](in_channels=config['in_channels'], out_channels=config['out_channels'], dropout=config['dropout'], batchnorm=config['batchnorm'], bias=config['bias'], pooling=config['pooling']).to(device)
    return model

# Training Loop 
def train(model, train_loader, test_loader, optimizer, criterion, device, epochs, save_name = None, save_path = None):
    """
    Train the given model on the training dataset and evaluate it on the test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion: Loss function used for training.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        epochs (int): Number of epochs for training.
        save_name (str, optional): Name of the file to save the trained model. Defaults to None.
        save_path (str, optional): Directory path to save the trained model file. Defaults to None.

    Returns:
        float: Final training loss.
    """
    epoch_loss = None # Store final training loss
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1:02d}/{epochs}', unit='batch') as pbar:
            for i, batch in enumerate(train_loader):
                inputs, targets = batch['image'].to(device), batch['mask'].to(device)                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()                
                epoch_loss += loss.item()
                pbar.update(1)
                # Display Test and Train Loss after each epoch
                if i + 1 == len(train_loader) :
                    # Calculate Test Loss after each epoch
                    test_loss = test(model, test_loader, criterion, device)
                    # Update Progress Bar with Test Loss
                    pbar.set_postfix({'Train Loss': epoch_loss / len(train_loader), 'Test Loss': test_loss})
                else :
                    # Update Progress Bar wuth only Epoch Loss
                    pbar.set_postfix({'Train Loss': epoch_loss / (i + 1), 'Test Loss' : '--'})
    # Calculate Final Losses
    test_loss = test(model, test_loader, criterion, device)
    print('--------------------------------------------------------------------------------------')
    print(f"Train Loss: {epoch_loss/len(train_loader)} | Test Loss: {test_loss}")
    if save_name is not None :
        torch.save(model.state_dict(), f"{os.path.join(save_path, save_name)}.pth")
    return epoch_loss

# Testing Loop
def test(model, test_loader, criterion, device):
    """
    Evaluate the given model on the test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion: Loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) on which to perform computations.

    Returns:
        float: Average loss on the test dataset.
    """
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for batch in test_loader:
            inputs, targets = batch['image'].to(device), batch['mask'].to(device)            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    return test_loss / len(test_loader)

# Evaluate the Model on Train and Test Sets and print the corresponding metric
def model_evaluator(model, train_loader, test_loader, device, config):
    """
    Evaluate a PyTorch model on training and test datasets and print the corresponding metrics.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        config (dict): A dictionary containing configuration parameters, including the task type (e.g., segmentation or regression).

    Returns:
        tuple: A tuple containing dictionaries of metrics for the training and test datasets.
    """
    # Check if the task type is segmentation or regression
    if config['task_type'].lower() == 'segmentation':
        table_title = f'--------------{type(model).__name__} - Segmentation Results---------------------'
        print(table_title)
        # Compute segmentation metrics for both training and test datasets
        train_metrics = segmentation_metrics(model, train_loader, device)
        test_metrics = segmentation_metrics(model, test_loader, device)
    else:
        table_title = f'----------------{type(model).__name__} - Regression Results---------------------'
        print(table_title)
        # Compute regression metrics for both training and test datasets
        train_metrics = regression_metrics(model, train_loader, device)
        test_metrics = regression_metrics(model, test_loader, device)
    # Print combined table for test and train metrics
    combined_metrics = [(k, train_metrics[k], test_metrics[k]) for k in train_metrics]
    headers = ["Metric", "Train", "Test"]
    print(tabulate(combined_metrics, headers=headers, tablefmt="grid"))
    print('-' * len(table_title))
    # Return test and train metrics for future use
    return train_metrics, test_metrics

# Model output Visualizer
def visualize_model_output(model, dataset, device, message = None):
    """
    Visualize the output of the model on random samples from the dataset.

    Args:
        model (torch.nn.Module): The trained model.
        dataset (torch.utils.data.Dataset): The dataset containing samples.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        message (str): The message to print before plotting.
    """
    if message is not None :
        # Print the message
        print(message)
    
    # Set the model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.inference_mode():
        # Create a figure with subplots
        fig, axes = plt.subplots(5, 3, figsize=(15, 15))

        for i in range(5):
            # Get random index within the dataset range
            index = random.randint(0, len(dataset) - 1)

            # Get random sample and label
            sample, label = dataset[index]['image'], dataset[index]['mask']

            # Extract input RGB and target DSMs from the sample and label
            input =  np.clip(sample[:3].cpu().numpy(), 0 ,1)
            target = label[0].cpu().numpy()

            # Get model output for the specified sample
            output = model(sample.unsqueeze(0).to(device))

            # Plot the Input RGB
            axes[i, 0].imshow(input.transpose(1, 2, 0))
            axes[i, 0].set_title("Input RGB")
            axes[i, 0].axis('off')

            # Plot the Target DSM
            axes[i, 1].imshow(target, cmap='gray')
            axes[i, 1].set_title("Target DSM")
            axes[i, 1].axis('off')

            # Plot the Output DSM
            axes[i, 2].imshow(output[0][0].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title("Output DSM")
            axes[i, 2].axis('off')

        # Adjust layout
        plt.tight_layout()
        plt.show()

# Function for hyperparameter tuning
def hyperparameter_tuning(dataset, device, config, n_trials=20):
    # Use only 10-20% of the dataset
    subset_size = int(0.1 * len(dataset))  # Adjust as needed
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset_dataset = Subset(dataset, subset_indices)
    # Define the hyperparameter search space
    def objective(trial):
        # Sample hyperparameters from the search space
        dropout = trial.suggest_float("dropout", 0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batchnorm = trial.suggest_categorical("batchnorm", [True, False])
        bias = trial.suggest_categorical("bias", [True, False])
        pooling = trial.suggest_categorical("pooling", ['maxpool', 'spectralpool'])
        # Initialize the model
        model = model_init(config)
        # Reassign Hyperparameters
        if config['composition'] == 0 :
            # Apply hyperparameters to the model
            if hasattr(model, 'dropout'):
                model.dropout = dropout
            if hasattr(model, 'batchnorm'):
                model.batchnorm = batchnorm
            if hasattr(model, 'bias'):
                model.bias = bias
            if hasattr(model, 'pooling'):
                model.pooling = pooling
        else :
            # For composition type models, reassign hyperparameters to regression_net
            if hasattr(model.regression_net, 'dropout'):
                model.regression_net.dropout = dropout
            if hasattr(model.regression_net, 'batchnorm'):
                model.regression_net.batchnorm = batchnorm
            if hasattr(model.regression_net, 'bias'):
                model.regression_net.bias = bias
            if hasattr(model.regression_net, 'pooling'):
                model.regression_net.pooling = pooling
        # Split subset dataset into train and test sets
        train_size = int(0.9 * len(subset_dataset))
        test_size = len(subset_dataset) - train_size
        train_dataset, test_dataset = random_split(subset_dataset, [train_size, test_size])
        # Loaders for train and test sets
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        # Define loss function and optimizer
        criterion = config['loss_fn']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        # Training loop
        for epoch in range(20):  # Number of Epochs
            model.train()
            for i, batch in enumerate(train_loader):
                inputs, targets = batch['image'].to(device), batch['mask'].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            # Evluate the model
            model.eval()
            if config['task_type'].lower() == 'regression' :
                metric = regression_metrics(model, test_loader, device)['delta1']
            else :
                metric = segmentation_metrics(model, test_loader, device)['f1_score'] 
            # Report intermediate result to Optuna
            trial.report(metric, epoch)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return metric
    # Create an Optuna study and optimize the objective function
    with tqdm(total=n_trials, desc="Optimizing", unit="trial") as pbar:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda study, trial: pbar.update()])
    # Get the best hyperparameters and accuracy
    best_params = study.best_params
    best_metric_val = study.best_value
    return best_params, best_metric_val

# Display Experiment Results
def display_experiment_results(evaluations, dataset_type, experiment_variable):
    """
    Display experiment results in a tabular format.

    Args:
        evaluations (dict): A dictionary containing evaluation metrics for different models.
        dataset_type (int): Training Set : 0 ; Test Set : 1
        experiment_variable(str): Name of the subject of experiment (eg Model, Loss Function)

    Returns:
        str: A string representing the formatted table of experiment results.
    """
    # Get the headers dynamically from the keys of the first dictionary in the evaluations
    headers = [experiment_variable] + list(evaluations[next(iter(evaluations))][dataset_type].keys())
    rows = []
    # Iterate over each model's evaluation metrics
    for variable_name, metrics in evaluations.items():
        row = [variable_name]
        # Populate rows with metric values for each model
        for metric_name in headers[1:]:
            row.append(metrics[dataset_type][metric_name.lower()])
        rows.append(row)
    # Highlight the maximum value in each column
    cols = list(zip(*rows))
    for col_idx in range(1, len(headers)):
        max_val = max(cols[col_idx])
        for row in rows:
            if row[col_idx] == max_val:
                row[col_idx] = f"\033[1m{row[col_idx]}\033[0m"
    # Return the formatted table of experiment results
    return tabulate(rows, headers=headers, tablefmt="grid")

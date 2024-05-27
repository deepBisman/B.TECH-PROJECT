# Handle Library Imports
import torch

# Segmentation Metrics
def segmentation_metrics(model, dataloader, device):
    """
    Calculate segmentation metrics for a given model and dataloader.

    Args:
        model (torch.nn.Module): The segmentation model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the test data.
        device (torch.device): The device to run the calculations on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: Dictionary containing average precision, recall, F1 score, and Dice coefficient.
    """
    # Initialize lists to store metrics for each image
    precision_list = []
    recall_list = []
    f1_score_list = []
    dice_coefficient_list = []

    # Set the model in evaluation mode
    model.eval()

    # Iterate through the test dataset
    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets = batch['image'].to(device), batch['mask'].to(device)

            # Forward pass through the model
            outputs = model(inputs)

            # Apply sigmoid activation to get probabilities
            probabilities = torch.sigmoid(outputs)

            # Threshold probabilities to obtain binary predictions (0 or 1)
            predictions = torch.round(probabilities)

            # Calculate true positives, true negatives, false positives, and false negatives
            true_positives = torch.sum((predictions == 1) & (targets == 1)).item()
            true_negatives = torch.sum((predictions == 0) & (targets == 0)).item()
            false_positives = torch.sum((predictions == 1) & (targets == 0)).item()
            false_negatives = torch.sum((predictions == 0) & (targets == 1)).item()

            # Calculate precision, recall, F1 score, and Dice coefficient for the current batch
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1_score_value = 2.0 * (precision * recall) / (precision + recall + 1e-10)
            dice_coefficient = (2.0 * true_positives) / (2.0 * true_positives + false_positives + false_negatives + 1e-10)

            # Append metrics to lists
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score_value)
            dice_coefficient_list.append(dice_coefficient)

    # Calculate average metrics over all batches
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1_score = sum(f1_score_list) / len(f1_score_list)
    avg_dice_coefficient = sum(dice_coefficient_list) / len(dice_coefficient_list)

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score,
        'dice_coefficient': avg_dice_coefficient
    }

def regression_metrics(model, dataloader, device):
    """
    Calculate regression metrics for a given model and dataloader.

    Args:
        model (torch.nn.Module): The regression model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the test data.
        device (torch.device): The device to run the calculations on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: Dictionary containing average MSE, RMSE, MAE, delta1, delta2, and delta3.
    """
    mse, rmse, mae = 0, 0, 0
    delta1, delta2, delta3 = 0, 0, 0

    # Set the model in evaluation mode
    model.eval()

    # Iterate through the test dataset
    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets = batch['image'].to(device), batch['mask'].to(device)

            # Forward pass through the model
            outputs = model(inputs)

            # Apply value constraints
            outputs[outputs == 0] = 0.00001
            outputs[outputs < 0] = 999
            targets[targets <= 0] = 0.00001

            # Create a valid mask
            valid_mask = ((targets > 0) + (outputs > 0)) > 0

            # Filter valid outputs and targets
            outputs = outputs[valid_mask]
            targets = targets[valid_mask]

            # Calculate absolute differences
            abs_diff = torch.abs(outputs - targets)

            # Update metrics
            mse += torch.mean(abs_diff ** 2).item()
            rmse += torch.sqrt((abs_diff ** 2).mean()).item()
            mae += torch.mean(abs_diff).item()

            max_ratio = torch.maximum(outputs / targets, targets / outputs)
            delta1 += torch.mean((max_ratio < 1.25).to(torch.float32)).item()
            delta2 += torch.mean((max_ratio < 1.25 ** 2).to(torch.float32)).item()
            delta3 += torch.mean((max_ratio < 1.25 ** 3).to(torch.float32)).item()

    # Average metrics over all batches
    num_batches = len(dataloader)
    mse /= num_batches
    rmse /= num_batches
    mae /= num_batches
    delta1 /= num_batches
    delta2 /= num_batches
    delta3 /= num_batches

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }


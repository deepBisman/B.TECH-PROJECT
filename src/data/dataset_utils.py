# Handle Library Imports
import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader, random_split

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Handle Module imports
from src.data.dataset_loader import CustomDataset

def calculate_channel_statistics(root_directory, num_channels):
    """
    Calculate the mean and standard deviation of each channel in the images found within the specified directory.

    Args:
        root_directory (str): Path to the directory containing the images.
        num_channels (int): Number of color channels in the images.

    Returns:
        tuple: Tuple containing lists of mean and standard deviation values for each channel.
    """
    channel_mean = torch.zeros(num_channels)
    channel_msq = torch.zeros(num_channels)
    total_images = 0
    # Loop Through each image
    for image_file in os.listdir(root_directory):
        image_path = os.path.join(root_directory, image_file)
        image = Image.open(image_path)
        image_tensor = transforms.ToTensor()(image)

        total_images += 1
        # Update mean values incrementally
        delta = image_tensor - channel_mean.view(num_channels, 1, 1) # Mean tensor of shape [3, 1, 1]
        channel_mean += delta.mean(dim=(1, 2)) / total_images 
        # Update squared mean values incrementally
        delta_squared = delta * (image_tensor - channel_mean.view(num_channels, 1, 1))
        channel_msq += delta_squared.mean(dim=(1, 2))

    # Calculate standard deviation for each channel
    channel_stddev = torch.zeros_like(channel_msq)
    if(total_images > 1):
        channel_stddev = torch.sqrt(channel_msq / (total_images - 1))
    # Return them as lists
    return channel_mean.tolist(), channel_stddev.tolist()

# Dataset Visualizer
def visualize_dataset(dataset, dataset_type='regression', num_samples=5):
    """
    Visualize random samples from the dataset.

    Args:
        dataset (CustomDataset): Dataset object.
        dataset_type (str): Type of dataset, either 'segmentation' or 'regression'. Defaults to 'regression'.
        num_samples (int): Number of samples to visualize. Defaults to 5.
    """
    # Get the maximum index for sampling
    max_index = len(dataset) - 1

    # Generate random indices for sampling
    random_indices = np.random.randint(0, max_index, size=num_samples)

    # Define a list of colors for the histogram
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

    # Iterate over random indices and visualize samples
    for index in random_indices:
        # Get sample and label from the dataset
        sample, label = dataset[index]['image'], dataset[index]['mask']

        # Extract RGB and SAR channels
        rgb_channels = np.clip(sample[:3].cpu().numpy(), 0, 1)
        sar_channel = sample[3].cpu().numpy()
        dsm = label[0].cpu().numpy()

        # Plot RGB image
        plt.figure(figsize=(20, 4))
        plt.subplot(141)
        plt.imshow(rgb_channels.transpose(1, 2, 0))  # Transpose channels for RGB
        plt.title("RGB Image")

        # Plot SAR image
        plt.subplot(142)
        plt.imshow(sar_channel, cmap='gray')  # Assuming SAR is single-channel (grayscale)
        plt.title("SAR Image")

        # Plot DSM
        plt.subplot(143)
        plt.imshow(dsm, cmap='gray')
        plt.title("DSM")

        # Plot histogram of DSM with random color and appropriate scale
        plt.subplot(144)
        random_color = random.choice(colors)
        
        if dataset_type.lower() == 'segmentation':
            # For segmentation, plot histogram from 0 to 1
            plt.hist(dsm.flatten(), bins=50, range=(0, 1), color=random_color, alpha=0.7)
            plt.title("DSM Histogram (0 to 1)")
        else:
            # For regression, plot histogram from 0 to 140 with log scale
            plt.hist(dsm.flatten(), bins=50, range=(0, 250), color=random_color, alpha=0.7, log=True)
            plt.title("DSM Histogram (Log Scale, 0 to 250)")

        plt.tight_layout()
        plt.show()

# Dataset Initializer
def dataset_init(config, seed=42):
    """
    Initialize dataset and dataloader objects

    Args:
        Config: Dictionery containing configuration of model
        seed : seed for RNG
    """
    # Check if normalization is on
    rgb_mean, rgb_std = None, None
    sar_mean,sar_std = None, None
    if config['normalization'] == 1:
        # Calculate Channel statistics for  Normalization
        rgb_mean, rgb_std = calculate_channel_statistics(os.path.join(config['data_path'],config['dataset_name'], 'rgb'), 3)
        sar_mean, sar_std = calculate_channel_statistics(os.path.join(config['data_path'],config['dataset_name'], 'sar'), 1)
    # Load dataset
    dataset = CustomDataset(os.path.join(config['data_path'], config['dataset_name']), config['dataset_type'], rgb_mean=rgb_mean, rgb_std=rgb_std, sar_mean=sar_mean, sar_std=sar_std)
    # Split dataset into train and test sets
    train_size = int(config['split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    # Load the Datatloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    return {
        'dataset' : dataset,
        'train_dataset' : train_dataset,
        'test_dataset' : test_dataset,
        'train_loader' : train_loader,
        'test_loader'  : test_loader,
    }

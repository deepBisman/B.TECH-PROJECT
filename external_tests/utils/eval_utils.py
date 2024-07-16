import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting tool
from torchvision import transforms
from PIL import Image

def generate_2d_null_mask(image_path):
    """
    Generate a 2D null mask from an image to identify pixels where all RGB channels are zero.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        torch.Tensor: A 2D tensor mask where pixels are 1 if all RGB channels are non-zero, else 0.
    """
    # Load input RGB image
    image = Image.open(image_path).convert('RGB')

    # Convert the image to a tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image)  # This will have shape [3, H, W]
    
    # Check if all channels are zero
    # We sum across the channel dimension and check if the sum is zero
    # This will give us a 2D tensor (mask) with values 0 where all channels are zero and >0 otherwise
    null_mask = torch.sum(image_tensor, dim=0) > 0
    
    # Convert the mask to integers (0 or 1)
    null_mask = null_mask.int().unsqueeze(dim=0)  
    
    return null_mask

def patchwise_inference(image_path, image_mean, image_stddev, model, device, patch_size = 512):
    """
    Process a large image by dividing it into patches, running them through a model,
    and reassembling the results.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        torch.Tensor: The output tensor containing the processed image.
    """
    input_image = Image.open(image_path).convert('RGB')

    # Define the transformation pipeline for normalization and conversion to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_stddev)
    ])
    
    # Apply the transformation
    input_tensor = transform(input_image).to(device)
    C, H, W = input_tensor.shape  # Get the dimensions of the image
    output_tensor = torch.zeros((1, H, W), dtype=torch.float32, device=device)  # Initialize output tensor
    
    # Process the image in patches
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            original_height = min(patch_size, H - i)
            original_width = min(patch_size, W - j)
            patch = input_tensor[:, i:i+patch_size, j:j+patch_size]

            # Pad the patch if it's smaller than the patch size
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                patch = torch.nn.functional.pad(patch, (0, patch_size - patch.shape[2], 0, patch_size - patch.shape[1]), mode='constant', value=0)
            
            patch = patch.unsqueeze(0)  # Add batch dimension

            # Perform inference on the patch
            with torch.inference_mode():  # Disable gradient calculation
                masked_output = model(patch)

            masked_output = masked_output.squeeze(0)  # Remove batch dimension

            # Copy the processed patch back to the corresponding location in the output tensor
            output_tensor[:, i:i+original_height, j:j+original_width] = masked_output[:, :original_height, :original_width]

    return output_tensor

def load_and_align_images(predicted_path, target_path, rgb_path):
    """
    Load and align images from raster files, crop to minimum dimensions, and apply multiple masks.

    Args:
        predicted_path (str): Path to the predicted raster image file.
        target_path (str): Path to the target raster image file.
        rgb_path (str): Path to the RGB image file used for generating the masks.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Tensor of the aligned predicted image.
            - torch.Tensor: Tensor of the aligned target image.
            - list: List of mask tensors, where each tensor identifies pixels where RGB channels are non-zero.
    """
    # Load images
    with rasterio.open(predicted_path) as predicted, rasterio.open(target_path) as target:
        predicted_img = predicted.read()
        target_img = target.read()

    # Convert to PyTorch tensors
    predicted_tensor = torch.from_numpy(predicted_img).float()
    target_tensor = torch.from_numpy(target_img).float()
    
    # Crop to the minimum dimension on either side
    min_height = min(predicted_tensor.size(1), target_tensor.size(1))
    min_width = min(predicted_tensor.size(2), target_tensor.size(2))
    predicted_tensor = predicted_tensor[:, :min_height, :min_width]
    target_tensor = target_tensor[:, :min_height, :min_width]
    
    # Generate Mask Tensors (example with null mask)
    null_mask_tensor = generate_2d_null_mask(rgb_path)

    # Create a list of mask tensors
    masks = [null_mask_tensor]  # List of masks
    
    # Apply all masks
    for i, mask_tensor in enumerate(masks):
        # Crop the mask tensors
        masks[i] = mask_tensor[:, :min_height, :min_width]
        # Mask the predicted tensor
        predicted_tensor = predicted_tensor * masks[i]

    return predicted_tensor, target_tensor, masks

def compute_metrics(predicted_tensor, target_tensor, masks):
    """
    Compute evaluation metrics based on predicted and target tensors after applying all masks.

    Args:
        predicted_tensor (torch.Tensor): Predicted tensor.
        target_tensor (torch.Tensor): Target tensor.
        masks (list): List of mask tensors indicating areas of interest.

    Returns:
        tuple: A tuple containing:
            - float: Delta1 metric (percentage of pixels with maxRatio < 1.25 after applying all masks).
            - float: Delta2 metric (percentage of pixels with maxRatio < 1.25^2 after applying all masks).
            - float: Delta3 metric (percentage of pixels with maxRatio < 1.25^3 after applying all masks).
    """
    # Ensure no division by zero or negative values
    predicted_tensor[predicted_tensor == 0] = 0.00001
    predicted_tensor[predicted_tensor < 0] = 999
    target_tensor[target_tensor <= 0] = 0.00001

    maxRatio = torch.maximum(predicted_tensor / target_tensor, target_tensor / predicted_tensor)
    
    # Combine all masks into a single mask
    combined_mask = torch.zeros_like(masks[0])
    for mask in masks:
        combined_mask = combined_mask | mask

    # Calculate metrics in areas of interest (masked areas)
    aoi = (masks[0] == 1)
    delta1_1 = (maxRatio < 1.25)
    delta2_1 = (maxRatio < (1.25 ** 2))
    delta3_1 = (maxRatio < (1.25 ** 3))
    
    delta1 = (delta1_1 & aoi).float().mean().item()
    delta2 = (delta2_1 & aoi).float().mean().item()
    delta3 = (delta3_1 & aoi).float().mean().item()

    return delta1, delta2, delta3

def plot_error_heatmap(predicted_tensor, target_tensor, masks):
    """
    Plot a heatmap of absolute percentage error between predicted and target raster images,
    masked by an RGB image.

    Args:
        predicted_tensor (str): Path to the predicted raster image file.
        target_tensor (str): Path to the target raster image file.
        masks (list): List of mask tensors indicating areas of interest.

    Returns:
        None
    """

    # Generate Combined Mask
    combined_mask = torch.zeros_like(masks[0])
    for mask in masks:
        combined_mask = combined_mask | mask
    
    # Apply Mask to Predicted Image & ensure non-divison by zero and penalties
    predicted_tensor = combined_mask * predicted_tensor
    predicted_tensor[predicted_tensor == 0] = 0.00001
    predicted_tensor[predicted_tensor < 0] = 999
    target_tensor[target_tensor <= 0] = 0.00001

    # Compute the absolute percentage error
    error = (np.maximum(target_tensor / predicted_tensor, predicted_tensor / target_tensor) - 1 ) * 100

    # Mask the error where combined_mask is 0 (no data)
    error_masked = np.where(combined_mask == 0, -1, error) # Use -1 for no-data areas

    # Define error ranges and corresponding colors (RGB normalized to [0, 1])
    error_ranges = [-1, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    colors = [
        (0.7, 0.9, 1),   # Very Light Blue, 0-5%
        (0.5, 0.8, 0.8), # Light Cyan, 5-10%
        (0.3, 0.7, 0.6), # Soft Green, 10-15%
        (0.2, 0.6, 0.3), # Green, 15-20%
        (1, 1, 0.2),     # Pale Yellow, 20-25%
        (1, 0.9, 0),     # Yellow, 25-30%
        (1, 0.8, 0),     # Light Orange, 30-35%
        (1, 0.6, 0),     # Orange, 35-40%
        (1, 0.4, 0),     # Dark Orange, 40-45%
        (1, 0.2, 0),     # Red-Orange, 45-50%
        (1, 0, 0),       # Red, 50-55%
        (0.9, 0, 0),     # Dark Red, 55-60%
        (0.8, 0, 0),     # Darker Red, 60-65%
        (0.7, 0, 0),     # Very Dark Red, 65-70%
        (0.6, 0, 0),     # Almost Black Red, 70-75%
        (0.5, 0, 0),     # Darker Red-Black, 75-80%
        (0.4, 0, 0),     # Near Black, 80-85%
        (0.3, 0, 0),     # Darker Near Black, 85-90%
        (0.2, 0, 0),     # Very Dark, 90-95%
        (0.1, 0, 0),     # Almost Black, 95-100%
        (0, 0, 0)        # Black, for errors > 100% (to catch any unexpected values)
    ]

    # Create a custom colormap
    cmap = ListedColormap(colors)

    # Categorize errors into the defined ranges
    error_bins = np.digitize(error_masked, bins=error_ranges) - 1

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    img = plt.imshow(error_bins, cmap=cmap, interpolation='nearest')

    # Adjust colorbar
    cbar = plt.colorbar(img, ticks=np.arange(len(error_ranges)))
    cbar_labels = ['No Data'] + [f'{error_ranges[i]}-{error_ranges[i+1]}%' for i in range(1, len(error_ranges)-1)]
    cbar_labels.append(f'>{error_ranges[-2]}%') # For errors above the last threshold
    cbar.ax.set_yticklabels(cbar_labels)  # Set custom labels
    cbar.set_label('Error Range (%)')

    plt.title('Error Heatmap')
    plt.axis('off')
    plt.show()

def scatter_plot_prediction_comparison_2D(prediction_tensor, target_tensor, masks):
    """
    Plot a heatmap of absolute percentage error between predicted and target raster images,
    masked by an RGB image.

    Args:
        predicted_tensor (str): Path to the predicted raster image file.
        target_tensor (str): Path to the target raster image file.
        masks (list): List of mask tensors indicating areas of interest.

    Returns:
        None
    """

    # Generate Combined Mask
    combined_mask = torch.zeros_like(masks[0])
    for mask in masks:
        combined_mask = combined_mask | mask
    
    # Apply Mask to Predicted Image & ensure non-divison by zero and penalties
    predicted_tensor = combined_mask * predicted_tensor
    predicted_tensor[predicted_tensor == 0] = 0.00001
    predicted_tensor[predicted_tensor < 0] = 999
    target_tensor[target_tensor <= 0] = 0.00001
    
    # Calculate absolute percentage error
    error = (np.maximum(target_tensor / predicted_tensor, predicted_tensor / target_tensor) - 1 ) * 100

    # Categorize based on percentage difference
    within_25 = (error <= 25) & (combined_mask != 0)
    within_55 = ((error > 25) & (error <= 55)) & (combined_mask != 0)
    within_95 = ((error > 55) & (error <= 95)) & (combined_mask != 0)
    beyond_95 = (error > 95) & (combined_mask != 0)
    non_data  = (combined_mask == 0)  # Non-data points
    
    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_tensor[within_25], target_tensor[within_25], color='lightblue', alpha=0.5, label='Within 25%')
    plt.scatter(predicted_tensor[within_55], target_tensor[within_55], color='blue', alpha=0.5, label='Within 55%')
    plt.scatter(predicted_tensor[within_95], target_tensor[within_95], color='red', alpha=0.5, label='Within 95%')
    plt.scatter(predicted_tensor[beyond_95], target_tensor[beyond_95], color='darkred', alpha=0.5, label='Beyond 95%')
    plt.scatter(predicted_tensor[non_data], target_tensor[non_data], color='grey', alpha=0.5, label='Non-data/Masked')
    
    plt.title('Comparison of Prediction vs Target Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Target Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_plot_prediction_comparison_3D(predicted_tensor, target_tensor, masks):
    """
    Plot a 3D scatter plot of predicted versus actual values, categorized by percentage difference,
    and excluding non-data/masked areas.

    Args:
        predicted_tensor (str): Path to the predicted raster image file.
        target_tensor (str): Path to the target raster image file.
        masks (list): List of mask tensors indicating areas of interest.

    Returns:
        None
    """
    # Generate Combined Mask
    combined_mask = torch.zeros_like(masks[0])
    for mask in masks:
        combined_mask = combined_mask | mask
    
    # Apply Mask to Predicted Image & ensure non-divison by zero and penalties
    predicted_tensor = combined_mask * predicted_tensor
    predicted_tensor[predicted_tensor == 0] = 0.00001
    predicted_tensor[predicted_tensor < 0] = 999
    target_tensor[target_tensor <= 0] = 0.00001
    
    # Calculate absolute percentage error
    error = (np.maximum(target_tensor / predicted_tensor, predicted_tensor / target_tensor) - 1 ) * 100

    # Categorize based on percentage difference
    within_25 = (error <= 25) & (combined_mask != 0)
    within_55 = ((error > 25) & (error <= 55)) & (combined_mask != 0)
    within_95 = ((error > 55) & (error <= 95)) & (combined_mask != 0)
    beyond_95 = (error > 95) & (combined_mask != 0)
    non_data  = (combined_mask == 0)  # Non-data points

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot categories with different colors
    ax.scatter(predicted_tensor[within_25], target_tensor[within_25], error[within_25], color='lightblue', alpha=0.5, label='Within 25%')
    ax.scatter(predicted_tensor[within_55], target_tensor[within_55], error[within_55], color='blue', alpha=0.5, label='Within 55%')
    ax.scatter(predicted_tensor[within_95], target_tensor[within_95], error[within_95], color='red', alpha=0.5, label='Within 95%')
    ax.scatter(predicted_tensor[beyond_95], target_tensor[beyond_95], error[beyond_95], color='darkred', alpha=0.5, label='Beyond 95%')
    ax.scatter(predicted_tensor[non_data], target_tensor[non_data], -1, color='grey', alpha=0.5, label='Non-data/Masked')
    
    # Set plot title and labels
    ax.set_title('3D Scatter Plot of Prediction vs Target Values')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Target Values')
    ax.set_zlabel('Percentage Difference')
    ax.legend()

    plt.show()





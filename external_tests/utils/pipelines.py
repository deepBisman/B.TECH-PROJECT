import os
import sys
import rasterio
import torch
import numpy as np
from rasterio.windows import Window, from_bounds
from tabulate import tabulate

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Dynamic Imports
from external_tests.utils.dataset_utils import resample_raster_bilinear, resample_raster_maxpool
from external_tests.utils.eval_utils import generate_2d_null_mask, patchwise_inference, load_and_align_images, compute_metrics

def crop_raster_to_bounds(input_raster_path, overlapping_bounds, output_raster_path):
    """
    Crop a raster image to the specified overlapping bounds and save as a new raster file.

    Args:
        input_raster_path (str): Path to the input raster image file.
        overlapping_bounds (tuple): Tuple containing (left, bottom, right, top) bounds to crop.
        output_raster_path (str): Path to save the cropped raster image file.
    
    """
    with rasterio.open(input_raster_path) as src:
        # Calculate the window to crop based on the overlapping bounds
        window = from_bounds(*overlapping_bounds, transform=src.transform)
        
        # Crop the image to the window
        cropped_image = src.read(window=window)
        
        # Define the new transform for the cropped image
        new_transform = src.window_transform(window)
        
        # Write the cropped image to a new file
        with rasterio.open(
            output_raster_path, 'w',
            driver='GTiff',
            height=window.height,
            width=window.width,
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=new_transform,
        ) as dst:
            dst.write(cropped_image)

def preprocessing_pipeline_5M(rgb_path, target_path, rgb_mean, rgb_stddev, model, device, patch_size=512, resampling_method='bilinear'):
    """
    A preprocessing pipeline for handling 3M to 5M image data, including resampling, cropping, inference,
    masking, and evaluation.

    Args:
        rgb_path (str): Path to the original RGB image file (3M resolution).
        target_path (str): Path to the target image file (e.g., ground truth).
        rgb_mean (tuple): Mean values for RGB normalization.
        rgb_stddev (tuple): Standard deviation values for RGB normalization.
        model (torch.nn.Module): The neural network model for inference.
        device (torch.device): Device (CPU or GPU) to perform inference.
        patch_size (int): Size of the patches for inference.
        resampling_method (str): Method for resampling ('bilinear' or 'maxpool').

    Returns:
        None
    """
    # Step 1) Downsample 3M RGB to 5M RGB
    downsampled_rgb_path = None
    if resampling_method == 'bilinear':
        dst = 'downsampled_rgb_image.tif'
        with rasterio.open(rgb_path) as src:
            with resample_raster_bilinear(input_raster=src, output_raster=dst, scale=0.6) as resampled:
                downsampled_rgb_path = dst
                print('Orig dims: {}, New dims: {}'.format(src.shape, resampled.shape))
                print(repr(resampled))
    else:
        dst = 'downsampled_rgb_image.tif'
        with rasterio.open(rgb_path) as src:
            with resample_raster_maxpool(input_raster=src, output_raster=dst, scale=0.6) as resampled:
                downsampled_rgb_path = dst
                print('Orig dims: {}, New dims: {}'.format(src.shape, resampled.shape))
                print(repr(resampled))
    # Step 2) Inference using the model on downsampled_rgb_img
    predicted_tensor = patchwise_inference(downsampled_rgb_path, rgb_mean, rgb_stddev, model, device, patch_size=512)
    
    # Generate Null Mask
    null_mask_tensor = generate_2d_null_mask(downsampled_rgb_path).to(device)
    
    # Mask the prediction
    predicted_tensor = predicted_tensor * null_mask_tensor
    predicted_tensor = predicted_tensor.squeeze().cpu().detach().numpy()
    
    # Open the original rgb image to access its transform and CRS
    with rasterio.open(downsampled_rgb_path) as src:
        transform = src.transform
        crs = src.crs
        # Prepare new metadata based on the original, but for the new data's shape and type
        new_meta = {
            'driver': 'GTiff',
            'height': predicted_tensor.shape[0],
            'width': predicted_tensor.shape[1],
            'count': predicted_tensor.shape[2] if predicted_tensor.ndim == 3 else 1,  # Bands
            'dtype': predicted_tensor.dtype,
            'crs': crs,
            'transform': transform
        }

    # Write the new image data with the updated metadata
    predicted_path = 'predicted.tif'
    with rasterio.open(predicted_path, 'w', **new_meta) as dst:
        if predicted_tensor.ndim == 2:  # For a single band (grayscale image)
            dst.write(predicted_tensor, 1)
        elif predicted_tensor.ndim == 3:  # For multi-band (e.g., RGB) images
            for i in range(predicted_tensor.shape[2]):
                dst.write(predicted_tensor[:, :, i], i + 1)

    # Step 3) Find Overlap bounds between predicted Image and target image
    cropped_predicted_path = 'cropped_predicted.tif'
    cropped_target_path = 'cropped_target.tif'
    cropped_downsampled_rgb_path = 'cropped_downsampled_rgb.tif'

    with rasterio.open(predicted_path) as predicted, rasterio.open(target_path) as target: 
        left = max(predicted.bounds.left, target.bounds.left)
        right = min(predicted.bounds.right, target.bounds.right)
        top = min(predicted.bounds.top, target.bounds.top)
        bottom = max(predicted.bounds.bottom, target.bounds.bottom)
    
        # Check for overlap
        if left < right and bottom < top:
            overlapping_bounds = (left, bottom, right, top)
            print("Overlapping area determined. Proceeding with cropping.")

            # Crop both images to the overlapping area and save them
            crop_raster_to_bounds(predicted_path, overlapping_bounds, cropped_predicted_path)
            crop_raster_to_bounds(target_path, overlapping_bounds, cropped_target_path)
            crop_raster_to_bounds(downsampled_rgb_path, overlapping_bounds, cropped_downsampled_rgb_path)

            print(f"Cropping completed. Images saved to {cropped_downsampled_rgb_path}, {cropped_predicted_path} and {cropped_target_path}")
        else:
            print("No overlapping area found. No cropping performed.")
            return
    
    # Step 4) Evaluate on target image
    predicted_tensor, target_tensor, masks = load_and_align_images(cropped_predicted_path, cropped_target_path, cropped_downsampled_rgb_path)
    delta1, delta2, delta3 = compute_metrics(predicted_tensor, target_tensor, masks)

    # Step 5) Tabulate results
    table = [
        ["Delta 1", delta1],
        ["Delta 2", delta2],
        ["Delta 3", delta3]
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt='grid'))

    # Step 6) Delete all created images
    cleanup_files = [downsampled_rgb_path, cropped_downsampled_rgb_path, cropped_target_path, predicted_path, cropped_predicted_path]
    for file_path in cleanup_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

def preprocessing_pipeline_3M(rgb_path, target_path, rgb_mean, rgb_stddev, model, device, patch_size=512, resampling_method='bilinear'):
    """
    A preprocessing pipeline for handling 3M to 5M image data, including resampling, cropping, inference,
    masking, and evaluation.

    Args:
        rgb_path (str): Path to the original RGB image file (3M resolution).
        target_path (str): Path to the target image file (e.g., ground truth).
        rgb_mean (tuple): Mean values for RGB normalization.
        rgb_stddev (tuple): Standard deviation values for RGB normalization.
        model (torch.nn.Module): The neural network model for inference.
        device (torch.device): Device (CPU or GPU) to perform inference.
        patch_size (int): Size of the patches for inference.
        resampling_method (str): Method for resampling ('bilinear' or 'maxpool').

    Returns:
        None
    """
    
    # Step 1) Inference using the model on rgb_img
    predicted_tensor = patchwise_inference(rgb_path, rgb_mean, rgb_stddev, model, device, patch_size=512)
    
    # Generate Null Mask
    null_mask_tensor = generate_2d_null_mask(rgb_path).to(device)
    
    # Mask the prediction
    predicted_tensor = predicted_tensor * null_mask_tensor
    predicted_tensor = predicted_tensor.squeeze().cpu().detach().numpy()
    
    # Open the original rgb image to access its transform and CRS
    with rasterio.open(rgb_path) as src:
        transform = src.transform
        crs = src.crs
        # Prepare new metadata based on the original, but for the new data's shape and type
        new_meta = {
            'driver': 'GTiff',
            'height': predicted_tensor.shape[0],
            'width': predicted_tensor.shape[1],
            'count': predicted_tensor.shape[2] if predicted_tensor.ndim == 3 else 1,  # Bands
            'dtype': predicted_tensor.dtype,
            'crs': crs,
            'transform': transform
        }
    
    # Write the new image data with the updated metadata
    predicted_path = 'predicted.tif'
    with rasterio.open(predicted_path, 'w', **new_meta) as dst:
        if predicted_tensor.ndim == 2:  # For a single band (grayscale image)
            dst.write(predicted_tensor, 1)
        elif predicted_tensor.ndim == 3:  # For multi-band (e.g., RGB) images
            for i in range(predicted_tensor.shape[2]):
                dst.write(predicted_tensor[:, :, i], i + 1)

    # Step 2) Find Overlap bounds between 3M Predicted Image and 5M Target image
    cropped_predicted_path = 'cropped_predicted.tif'
    cropped_target_path = 'cropped_target.tif'
    cropped_rgb_path = 'cropped_rgb.tif'

    with rasterio.open(predicted_path) as predicted, rasterio.open(target_path) as target: 
        left = max(predicted.bounds.left, target.bounds.left)
        right = min(predicted.bounds.right, target.bounds.right)
        top = min(predicted.bounds.top, target.bounds.top)
        bottom = max(predicted.bounds.bottom, target.bounds.bottom)
    
        # Check for overlap
        if left < right and bottom < top:
            overlapping_bounds = (left, bottom, right, top)
            print("Overlapping area determined. Proceeding with cropping.")

            # Crop both images to the overlapping area and save them
            crop_raster_to_bounds(predicted_path, overlapping_bounds, cropped_predicted_path)
            crop_raster_to_bounds(target_path, overlapping_bounds, cropped_target_path)
            crop_raster_to_bounds(rgb_path, overlapping_bounds, cropped_rgb_path)
            
            print(f"Cropping completed. Images saved to {cropped_rgb_path}, {cropped_predicted_path} and {cropped_target_path}")
        else:
            print("No overlapping area found. No cropping performed.")
            return
    
    # Step 3) Downsample 3M RGB Prediction to 5M RGB Prediction
    downsampled_cropped_predicted_path = None
    if resampling_method == 'bilinear':
        dst = 'downsampled_cropped_predicted_image.tif'
        with rasterio.open(cropped_predicted_path) as src:
            with resample_raster_bilinear(input_raster=src, output_raster=dst, scale=0.6) as resampled:
                downsampled_cropped_predicted_path = dst
                print('Orig dims: {}, New dims: {}'.format(src.shape, resampled.shape))
                print(repr(resampled))
    else:
        dst = 'downsampled_cropped_predicted_image.tif'
        with rasterio.open(cropped_predicted_path) as src:
            with resample_raster_maxpool(input_raster=src, output_raster=dst, scale=0.6) as resampled:
                downsampled_cropped_predicted_path = dst
                print('Orig dims: {}, New dims: {}'.format(src.shape, resampled.shape))
                print(repr(resampled))
    
    # Step 4) Evaluate on target image
    predicted_tensor, target_tensor, masks = load_and_align_images(downsampled_cropped_predicted_path, cropped_target_path, cropped_rgb_path)
    delta1, delta2, delta3 = compute_metrics(predicted_tensor, target_tensor, masks)

    # Step 5) Tabulate results
    table = [
        ["Delta 1", delta1],
        ["Delta 2", delta2],
        ["Delta 3", delta3]
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt='grid'))

    # Step 6) Delete all created images
    cleanup_files = [cropped_rgb_path, cropped_target_path, predicted_path, cropped_predicted_path, downsampled_cropped_predicted_path]
    for file_path in cleanup_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

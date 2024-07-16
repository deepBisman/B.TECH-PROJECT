import rasterio
import numpy as np
import os
from tqdm import tqdm
from rasterio.warp import transform
from rasterio.enums import Resampling
from rasterio import Affine, MemoryFile
from contextlib import contextmanager
from tabulate import tabulate

def analyze_raster_image(image_path, print_as_table=False):
    """
    Analyze a raster image and extract geographical and statistical information.

    Args:
        image_path (str): Path to the raster image file.
        print_as_table (bool): If True, print the information as a table.

    Returns:
        dict: A dictionary containing the following information:
            - 'top_left_coordinates' (tuple): Coordinates (latitude, longitude) of the top left corner.
            - 'bottom_right_coordinates' (tuple): Coordinates (latitude, longitude) of the bottom right corner.
            - 'dimensions' (tuple): Dimensions (width, height) of the image.
            - 'mean_pixel_value' (float): Mean value of the pixels in the first band.
            - 'std_dev_pixel_value' (float): Standard deviation of the pixel values in the first band.
            - 'pixel_value_range' (tuple): Minimum and maximum pixel values in the first band.
    """
    with rasterio.open(image_path) as src:
        # Extract the bounds (in image's CRS)
        bounds = src.bounds
        # Convert bounds from the image's CRS to WGS84 (Lat/Lon)
        bottom_left_latlon = transform(src.crs, {'init': 'EPSG:4326'}, [bounds.left], [bounds.bottom])
        top_right_latlon = transform(src.crs, {'init': 'EPSG:4326'}, [bounds.right], [bounds.top])
        
        # Image size
        width, height = src.width, src.height
        
        # Read the first band (assuming a single-band image for simplicity)
        band1 = src.read(1)
        
        # Calculate mean, standard deviation, and pixel range
        mean_val = band1.mean()
        std_dev = band1.std()
        pixel_min, pixel_max = band1.min(), band1.max()
    
    result = {
        "top_left_coordinates": (top_right_latlon[1][0], bottom_left_latlon[0][0]),
        "bottom_right_coordinates": (bottom_left_latlon[1][0], top_right_latlon[0][0]),
        "dimensions": (width, height),
        "mean_pixel_value": mean_val,
        "std_dev_pixel_value": std_dev,
        "pixel_value_range": (pixel_min, pixel_max)
    }

    if print_as_table:
        table = [
            ["Top Left Coordinates (Latitude, Longitude)", result["top_left_coordinates"]],
            ["Bottom Right Coordinates (Latitude, Longitude)", result["bottom_right_coordinates"]],
            ["Image Dimensions (Width, Height)", result["dimensions"]],
            ["Mean Pixel Value", result["mean_pixel_value"]],
            ["Pixel Value Standard Deviation", result["std_dev_pixel_value"]],
            ["Pixel Value Range (Min, Max)", result["pixel_value_range"]]
        ]
        print(tabulate(table, headers=["Property", "Value"], tablefmt="pretty"))
    
    return result

@contextmanager
def resample_raster_bilinear(input_raster, output_raster, scale=2):
    """
    Resample a raster image using bilinear interpolation.

    Args:
        input_raster (rasterio.io.DatasetReader): The input raster dataset.
        output_raster (str): Path to the output raster file.
        scale (float): The scaling factor for resampling (e.g., 3m to 5m is 0.6).
    """
    t = input_raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = input_raster.height * scale
    width = input_raster.width * scale

    profile = input_raster.profile
    profile.update(transform=transform, driver='GTiff', height=int(height), width=int(width))

    data = input_raster.read(
            out_shape=(input_raster.count, int(height), int(width)),
            resampling=Resampling.bilinear,
        )

    # Write to the output file path
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
        with rasterio.open(output_raster, 'w', **profile) as out_dataset:
            out_dataset.write(data)
            del data
        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return
    
            
@contextmanager
def resample_raster_maxpool(input_raster, output_raster, scale=2):
    """
    Resample a raster image using max pooling.

    Args:
        input_raster (rasterio.io.DatasetReader): The input raster dataset.
        output_raster (str): Path to the output raster file.
        scale (float): The scaling factor for resampling (e.g., 3m to 5m is 0.6).
    """
    t = input_raster.transform
    # Adjust the metadata for downsampling
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = input_raster.height * scale
    width = input_raster.width * scale

    profile = input_raster.profile
    profile.update(transform=transform, driver='GTiff', height=int(height), width=int(width))

    # Initialize an empty array for the downsampled data
    downsampled_data = np.zeros((input_raster.count, int(height), int(width)), dtype=input_raster.dtypes[0])

    # Read the full-resolution data
    data = input_raster.read()

    # Manually implement max pooling for downsampling
    for band in range(data.shape[0]):
        for i in range(int(height)):
            for j in range(int(width)):
                # Extract the current window
                window = data[band, int(i/scale):int((i+1)/scale), int(j/scale):int((j+1)/scale)]
                # Find the max value in the window
                downsampled_data[band, i, j] = np.max(window)

    # Write to the output file path
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(downsampled_data)
        with rasterio.open(output_raster, 'w', **profile) as out_dataset:
            out_dataset.write(downsampled_data)
            del downsampled_data  # Free memory
        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return

def resample_directory(input_dir, output_dir, scale, flag=False):
    """
    Resample all raster images in the input directory using the specified method
    and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing input raster images.
        output_dir (str): Path to the directory to save the resampled raster images.
        scale (float): The scaling factor for resampling.
        flag (bool): If True, use bilinear interpolation; if False, use max-pooling.
    """
    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of files to process
    files = [f for f in os.listdir(input_dir) if f.endswith(('.tif', '.tiff'))]
    
    # Resample the dataset with progress bar
    for filename in tqdm(files, desc="Resampling rasters"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with rasterio.open(input_path) as src:
            if flag:
                with resample_raster_bilinear(src, output_path, scale) as resampled:
                    pass
            else:
                with resample_raster_maxpool(src, output_path, scale) as resampled:
                    pass
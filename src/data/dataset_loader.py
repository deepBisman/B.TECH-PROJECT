# Handle Imports
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

# Custom Normalizer
class ConditionalNormalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        if self.mean is not None and self.std is not None:
            self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        else:
            self.normalize = None

    def __call__(self, img):
        if self.normalize:
            return self.normalize(img)
        return img
    
# Custom Dataloader
class CustomDataset(Dataset):
    def __init__(self, root_path, dataset_type='regression', rgb_mean=None, rgb_std=None, sar_mean=None, sar_std=None):
        """
        Custom dataset class for loading RGB/SAR images.

        Args:
            root_path (str): Path to the dataset directory.
            dataset_type (str): Type of dataset, either 'regression' or 'segmentation'.
            rgb_mean (list, optional): Mean values for RGB images. Defaults to None.
            rgb_std (list, optional): Standard deviation values for RGB images. Defaults to None.
            sar_mean (list, optional): Mean values for SAR images. Defaults to None.
            sar_std (list, optional): Standard deviation values for SAR images. Defaults to None.
        """
        # Define paths to the RGB, SAR, and DSM directories
        self.rgb_path = os.path.join(root_path, "rgb")
        self.sar_path = os.path.join(root_path, "sar")
        self.dsm_path = os.path.join(root_path, "dsm")

        # Get the list of files in each directory and sort them
        self.rgb_files = os.listdir(self.rgb_path)
        self.sar_files = os.listdir(self.sar_path)
        self.dsm_files = os.listdir(self.dsm_path)
        self.rgb_files.sort()
        self.sar_files.sort()
        self.dsm_files.sort()

        # Store the length of the dataset (number of images in RGB directory)
        self.length = len(self.rgb_files)

        # Define transformations for RGB and SAR images
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            ConditionalNormalize(mean=rgb_mean, std=rgb_std),  # Normalize RGB images
        ])
        self.sar_transform = transforms.Compose([
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            ConditionalNormalize(mean=sar_mean, std=sar_std),  # Normalize SAR images
        ])

        # Define transformation for DSM images based on dataset type
        if dataset_type.lower() == 'segmentation':
            self.dsm_transform = transforms.Compose([
                transforms.ToTensor(), # Convert the image to a PyTorch tensor
                transforms.Lambda(lambda x: (x > 0).type(x.dtype) * 1),  # Convert DSM to binary
            ])
        else:
            self.dsm_transform = transforms.ToTensor()  # No transformation for regression

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Get item from dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing RGB/SAR image and DSM image.
        """
        # Open RGB, SAR, and DSM images
        rgb_image = Image.open(os.path.join(self.rgb_path, self.rgb_files[index]))
        sar_image = Image.open(os.path.join(self.sar_path, self.sar_files[index]))
        dsm_image = Image.open(os.path.join(self.dsm_path, self.dsm_files[index]))

        # Transform RGB and SAR images separately
        rgb_image = self.rgb_transform(rgb_image)
        sar_image = self.sar_transform(sar_image)

        # Perform DSM transformation if segmentation is selected
        dsm_image = self.dsm_transform(dsm_image)

        return {'image' : torch.cat((rgb_image, sar_image), dim=0), 'mask' : dsm_image}
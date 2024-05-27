# Handle Library Imports
#import cupy as cp
import sigpy as sp
import numpy as np

def spectral_crop(array, array_shape, bounding_shape):
    """
    Crop the input array to match the specified bounding shape.

    Args:
        array (cupy.ndarray): Input array to be cropped.
        array_shape (tuple): Shape of the input array.
        bounding_shape (tuple): Desired shape after cropping.

    Returns:
        cupy.ndarray: Cropped array.
    """
    start = tuple((a - da) // 2 for a, da in zip(array_shape, bounding_shape))
    end = tuple(s + de for s, de in zip(start, bounding_shape))
    slices = tuple(slice(start[i], end[i]) for i in range(len(start)))
    return array[slices]

def spectral_pad(array, array_shape, bounding_shape):
    """
    Pad the input array to match the specified bounding shape.

    Args:
        array (cupy.ndarray): Input array to be padded.
        array_shape (tuple): Shape of the input array.
        bounding_shape (tuple): Desired shape after padding.

    Returns:
        cupy.ndarray: Padded array.
    """
    out = np.zeros(bounding_shape)
    start = tuple((b - da) // 2 for b, da in zip(bounding_shape, array_shape))
    end = tuple(s + de for s, de in zip(start, array_shape))
    slices = tuple(slice(start[i], end[i]) for i in range(len(start)))
    out[slices] = array
    return out

def discrete_hartley_transform(input):
    """
    Compute the discrete Hartley transform of the input array.

    Args:
        input (cupy.ndarray): Input array.

    Returns:
        cupy.ndarray: Transformed array.
    """
    N = input.ndim
    axes_n = np.arange(2, N)
    fft = sp.fft(input, axes=axes_n)
    H = fft.real - fft.imag
    return H

def crop_forward(input, return_shape):
    """
    Forward pass of spectral crop operation.

    Args:
        input (cupy.ndarray): Input array.
        return_shape (tuple): Desired shape after cropping.

    Returns:
        cupy.ndarray: Cropped array.
    """
    output_shape = np.zeros(input.ndim, dtype=int)
    output_shape[0] = input.shape[0]
    output_shape[1] = input.shape[1]
    output_shape[2:] = np.asarray(return_shape, dtype=int)
    dht = discrete_hartley_transform(input)
    dht = spectral_crop(dht, dht.shape, output_shape)
    dht = discrete_hartley_transform(dht)
    return dht

def pad_backward(grad_output, input_shape):
    """
    Backward pass of spectral pad operation.

    Args:
        grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        input_shape (tuple): Shape of the input array.

    Returns:
        cupy.ndarray: Padded gradient array.
    """
    dht = discrete_hartley_transform(grad_output)
    dht = spectral_pad(dht, dht.shape, input_shape)
    dht = discrete_hartley_transform(dht)
    return dht

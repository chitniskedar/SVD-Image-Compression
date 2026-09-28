"""
Image utilities for the SVD Image Compression project.

This file handles the basic image processing part:
    1. Load an image
    2. Convert it to grayscale
    3. Convert it into a matrix of pixel values

For a grayscale image, every pixel is represented by a value
between 0 and 255.

So an image of size m x n can be represented as a matrix:

        A = [a11 a12 ... a1n
             a21 a22 ... a2n
             .
             .
             .
             am1 am2 ... amn]

This matrix A is what we pass to SVD.
"""

from PIL import Image
import numpy as np


def load_image(path):
    """
    Load the image from the given path.

    We convert the image to grayscale because it makes the
    Linear Algebra part much simpler.

    Instead of dealing with 3 matrices (R, G, B), we only
    have one matrix containing pixel intensities.

    Returns:
        image: PIL Image object
        matrix: NumPy array containing pixel values
    """

    image = Image.open(path).convert("L")

    # Convert the image into a NumPy matrix.
    # Each element represents the intensity of one pixel.
    matrix = np.array(image, dtype=np.float64)

    return image, matrix


def save_image(matrix, path):
    """
    Convert a matrix back into an image and save it.

    SVD reconstruction can sometimes produce values slightly
    below 0 or above 255, so we clip them before saving.

    0   -> black
    255 -> white
    """

    matrix = np.clip(matrix, 0, 255)

    # Convert floating point values back to 8-bit pixel values.
    image = Image.fromarray(matrix.astype(np.uint8))

    image.save(path)
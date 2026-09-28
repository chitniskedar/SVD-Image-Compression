"""
Main program for SVD Image Compression.

This file connects all the modules together.

Pipeline:

    Input Image
        ↓
    Convert to Matrix
        ↓
    SVD
        ↓
    Select k
        ↓
    Rank-k Approximation
        ↓
    Reconstructed Image
        ↓
    Save Output

We test multiple values of k to observe the trade-off
between compression and image quality.
"""

import os

from image_utils import load_image, save_image
from svd_compression import apply_svd, reconstruct_image


# Input image
INPUT_PATH = "../input/sample.jpg"

# Where reconstructed images will be stored
OUTPUT_DIR = "../output"

# Different numbers of singular values we want to test
K_VALUES = [5, 20, 50, 100]


def main():
    """
    Run the complete SVD compression pipeline.
    """

    # Create output directory if it doesn't already exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Load the image
    # ---------------------------------------------------------

    image, matrix = load_image(INPUT_PATH)

    print("Image loaded successfully.")
    print(f"Image dimensions: {matrix.shape}")

    # ---------------------------------------------------------
    # STEP 2: Apply SVD
    # ---------------------------------------------------------

    print("\nApplying SVD...")

    U, S, VT = apply_svd(matrix)

    print("SVD completed.")

    # Print some information about the decomposition.
    print(f"Number of singular values: {len(S)}")

    # ---------------------------------------------------------
    # STEP 3: Reconstruct image for different k values
    # ---------------------------------------------------------

    for k in K_VALUES:

        # We cannot use a k larger than the number of
        # available singular values.
        if k > len(S):
            print(f"\nSkipping k={k}: not enough singular values.")
            continue

        print(f"\nReconstructing image using k={k}...")

        reconstructed = reconstruct_image(U, S, VT, k)

        # Output filename
        output_path = os.path.join(
            OUTPUT_DIR,
            f"compressed_k{k}.png"
        )

        # Save reconstructed image
        save_image(reconstructed, output_path)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
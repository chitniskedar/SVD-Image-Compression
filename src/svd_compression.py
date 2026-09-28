"""
SVD based image compression.

The main idea of the project is:

                A = U Σ Vᵀ

where:
    A   -> original image matrix
    U   -> left singular vectors
    Σ   -> singular values
    Vᵀ  -> transpose of right singular vectors

Instead of keeping all singular values, we keep only the
largest k singular values.

This gives us a low-rank approximation:

                A_k = U_k Σ_k V_kᵀ

The smaller k is, the more we compress the image.
The larger k is, the more information we retain.
"""

import numpy as np


def apply_svd(matrix):
    """
    Perform Singular Value Decomposition on the image matrix.

    NumPy decomposes A as:

                    A = U Σ Vᵀ

    Returns:
        U  -> left singular vectors
        S  -> singular values
        VT -> transpose of right singular vectors
    """

    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    return U, S, VT


def reconstruct_image(U, S, VT, k):
    """
    Reconstruct the image using only the first k singular values.

    Normally we would reconstruct the entire matrix using:

                    A = U Σ Vᵀ

    But for compression we only use:

                    A_k = U_k Σ_k V_kᵀ

    where k is the number of singular values we keep.

    The first k singular values are the largest and generally
    contain the most important information about the image.

    Example:

        k = 5
        -> keep only first 5 singular values

        k = 100
        -> keep first 100 singular values

    Returns:
        reconstructed image matrix
    """

    # Keep only the first k columns of U.
    U_k = U[:, :k]

    # Keep only the first k singular values.
    S_k = S[:k]

    # Keep only the first k rows of Vᵀ.
    VT_k = VT[:k, :]

    # Σ is a diagonal matrix containing singular values.
    # np.diag(S_k) creates that diagonal matrix.
    Sigma_k = np.diag(S_k)

    # Low-rank approximation:
    #
    #       A_k = U_k Σ_k V_kᵀ
    #
    reconstructed = U_k @ Sigma_k @ VT_k

    return reconstructed


def compress_image(matrix, k):
    """
    Convenience function that performs the complete SVD
    compression process for a particular k.

    Steps:
        1. Perform SVD
        2. Keep first k singular values
        3. Reconstruct the image
    """

    U, S, VT = apply_svd(matrix)

    reconstructed = reconstruct_image(U, S, VT, k)

    return reconstructed, S
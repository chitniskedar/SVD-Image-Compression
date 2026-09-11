"""Linear Algebra core for grayscale SVD image compression.

Person 1's module represents an image as a matrix ``A`` and computes
``A = U Sigma V^T``.  Retaining only the largest k singular components gives
the low-rank approximation ``A_k = U_k Sigma_k V_k^T``.  Its theoretical SVD
representation has ``k(m + n + 1)`` values, not necessarily that file-size
ratio after PNG/JPEG encoding. Image I/O, metrics, graphs, and UI live elsewhere.
"""
from __future__ import annotations

import numpy as np


def _valid_k(k: int, maximum: int) -> int:
    """Validate an integer component count used internally by this module."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= int(k) <= maximum:
        raise ValueError(f"k must be an integer between 1 and {maximum}. Received k={k}.")
    return int(k)


def perform_svd(image_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform the compact SVD decomposition ``A = U Sigma V^T`` once per image.

    Parameters
    ----------
    image_matrix : numpy.ndarray
        Finite numeric 2D grayscale matrix ``A`` of shape ``(m, n)``; rows and
        columns represent pixel positions. Pixels usually range from 0 to 255.

    Returns
    -------
    U : numpy.ndarray
        Left singular vectors, shape ``(m, r)``.
    singular_values : numpy.ndarray
        Descending non-negative diagonal values of Sigma, shape ``(r,)``.
    Vt : numpy.ndarray
        Transposed right singular vectors, shape ``(r, n)``; ``r=min(m,n)``.

    Raises
    ------
    ValueError
        If the input is not a non-empty finite numeric 2D NumPy array, or SVD fails.

    Notes
    -----
    ``full_matrices=False`` produces the compact SVD, which contains every
    component required for rank-k reconstruction without unnecessary vectors.

    Examples
    --------
    >>> U, S, Vt = perform_svd(np.array([[10., 20.], [30., 40.]]))
    >>> S.shape
    (2,)
    """
    if not isinstance(image_matrix, np.ndarray) or image_matrix.ndim != 2 or image_matrix.size == 0:
        raise ValueError("image_matrix must be a non-empty 2D NumPy array.")
    if not np.issubdtype(image_matrix.dtype, np.number) or not np.isfinite(image_matrix).all():
        raise ValueError("image_matrix must contain finite numeric values.")
    try:
        return np.linalg.svd(image_matrix.astype(float, copy=False), full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("SVD could not be computed for this image matrix.") from error


def reconstruct_image(U: np.ndarray, singular_values: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """Create a clipped uint8 rank-k reconstruction, ``A_k = U_k Sigma_k V_k^T``.

    Parameters
    ----------
    U : numpy.ndarray
        Left singular-vector matrix from :func:`perform_svd`, shape ``(m, r)``.
    singular_values : numpy.ndarray
        Descending singular values from SVD, shape ``(r,)``.
    Vt : numpy.ndarray
        Transpose of right singular vectors from SVD, shape ``(r, n)``.
    k : int
        Number of largest singular values and matching vectors to keep;
        must satisfy ``1 <= k <= r``.

    Returns
    -------
    numpy.ndarray
        The rank-k approximation, shape ``(m, n)``, clipped to [0, 255] and
        converted to ``numpy.uint8`` for later image handling.

    Raises
    ------
    ValueError
        If decomposition arrays are incompatible or k is invalid.

    Notes
    -----
    ``U_k`` uses U's first k columns, ``Sigma_k`` contains the first k values,
    and ``V_k^T`` uses Vt's first k rows. Increasing k preserves more matrix
    information but requires more stored values.

    Examples
    --------
    >>> U, S, Vt = perform_svd(np.array([[10., 20.], [30., 40.]]))
    >>> reconstruct_image(U, S, Vt, 1).dtype
    dtype('uint8')
    """
    if not all(isinstance(x, np.ndarray) for x in (U, singular_values, Vt)):
        raise ValueError("U, singular_values, and Vt must be NumPy arrays.")
    if U.ndim != 2 or singular_values.ndim != 1 or Vt.ndim != 2:
        raise ValueError("U and Vt must be 2D; singular_values must be 1D.")
    r = singular_values.size
    if r == 0 or U.shape[1] != r or Vt.shape[0] != r:
        raise ValueError("SVD arrays have incompatible component dimensions.")
    k = _valid_k(k, r)
    # Explicit truncated-SVD mathematics for the viva.
    U_k = U[:, :k]                 # first k left singular vectors
    S_k = singular_values[:k]      # k largest singular values
    Sigma_k = np.diag(S_k)         # k x k diagonal matrix
    Vt_k = Vt[:k, :]               # matching right singular vectors transposed
    reconstructed = U_k @ Sigma_k @ Vt_k
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def calculate_storage_elements(rows: int, cols: int, k: int) -> int:
    """Return numerical values required by the theoretical rank-k SVD form.

    Parameters
    ----------
    rows : int
        Positive number ``m`` of original matrix rows.
    cols : int
        Positive number ``n`` of original matrix columns.
    k : int
        Retained component count from 1 through ``min(rows, cols)``.

    Returns
    -------
    int
        ``rows*k + k + k*cols``, equivalently ``k(rows + cols + 1)``.

    Raises
    ------
    ValueError
        If dimensions are not positive integers or k is invalid.

    Notes
    -----
    The terms respectively store U_k, Sigma_k's diagonal, and V_k^T.

    Examples
    --------
    >>> calculate_storage_elements(512, 512, 20)
    20500
    """
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0 for x in (rows, cols)):
        raise ValueError("rows and cols must be positive integers.")
    k = _valid_k(k, min(int(rows), int(cols)))
    return k * (int(rows) + int(cols) + 1)


def calculate_compression_ratio(rows: int, cols: int, k: int) -> float:
    """Return the theoretical SVD representation compression ratio.

    Parameters
    ----------
    rows, cols : int
        Positive dimensions of the original image matrix.
    k : int
        Number of singular components retained.

    Returns
    -------
    float
        ``(rows*cols) / [k(rows + cols + 1)]``: original values divided by
        theoretical truncated-SVD representation values.

    Raises
    ------
    ValueError
        If inputs fail :func:`calculate_storage_elements` validation.

    Notes
    -----
    This measures mathematical storage only; it is not an actual PNG/JPEG
    file-size ratio.

    Examples
    --------
    >>> round(calculate_compression_ratio(512, 512, 20), 2)
    12.79
    """
    return (int(rows) * int(cols)) / calculate_storage_elements(rows, cols, k)


def get_singular_value_information(singular_values: np.ndarray, count: int = 10) -> np.ndarray:
    """Return a copy of the leading singular values for a future report or UI.

    Parameters
    ----------
    singular_values : numpy.ndarray
        One-dimensional non-empty numeric SVD output array. Its largest values
        correspond to the most important matrix components.
    count : int, default=10
        Number of leading values requested, from 1 through the array length.

    Returns
    -------
    numpy.ndarray
        Independent 1D array containing the first ``count`` singular values.

    Raises
    ------
    ValueError
        If the array is invalid or count is out of range.

    Examples
    --------
    >>> get_singular_value_information(np.array([9., 4., 1.]), 2)
    array([9., 4.])
    """
    if not isinstance(singular_values, np.ndarray) or singular_values.ndim != 1 or singular_values.size == 0:
        raise ValueError("singular_values must be a non-empty 1D NumPy array.")
    if not np.issubdtype(singular_values.dtype, np.number):
        raise ValueError("singular_values must contain numeric values.")
    return singular_values[:_valid_k(count, singular_values.size)].copy()


def calculate_energy_retention(singular_values: np.ndarray, k: int) -> float:
    """Return retained squared-singular-value energy as a percentage.

    Parameters
    ----------
    singular_values : numpy.ndarray
        One-dimensional non-empty numeric array returned by SVD.
    k : int
        Number of leading singular values retained.

    Returns
    -------
    float
        ``sum(S[:k]**2) / sum(S**2) * 100``. A zero-energy matrix returns 100.

    Raises
    ------
    ValueError
        If singular_values or k is invalid.

    Notes
    -----
    Squared singular values give the Frobenius-norm energy contribution of
    each component, explaining why leading components often approximate well.

    Examples
    --------
    >>> round(calculate_energy_retention(np.array([3., 4.]), 1), 1)
    36.0
    """
    values = get_singular_value_information(singular_values, singular_values.size)
    k = _valid_k(k, values.size)
    total = float(np.sum(values ** 2))
    return 100.0 if total == 0 else float(np.sum(values[:k] ** 2) / total * 100)


__all__ = ["perform_svd", "reconstruct_image", "calculate_storage_elements",
           "calculate_compression_ratio", "get_singular_value_information",
           "calculate_energy_retention"]

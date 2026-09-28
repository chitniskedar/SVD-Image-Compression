"""metrics.py  --  Person 2 (Modules 5, 6, 7)

Compression size, compression ratio, MSE and PSNR for SVD image compression.
"""
import os
import numpy as np


# ---------- Module 5: Compression size ----------
def original_size(m: int, n: int) -> int:
    """Number of values needed to store the full m x n matrix."""
    return m * n


def compressed_size(m: int, n: int, k: int) -> int:
    """Rank-k SVD storage: U_k (m*k) + Sigma_k (k) + V_k (n*k) = k(m+n+1)."""
    return k * (m + n + 1)


def compression_ratio(m: int, n: int, k: int) -> float:
    """mn / k(m+n+1).  A value < 1 means the SVD form is LARGER than the original."""
    return original_size(m, n) / compressed_size(m, n, k)


def storage_saved_percent(m: int, n: int, k: int) -> float:
    """Percentage of storage saved (negative if the SVD form is bigger)."""
    return (1 - compressed_size(m, n, k) / original_size(m, n)) * 100


def break_even_rank(m: int, n: int) -> float:
    """Largest k for which SVD storage is still smaller than mn: k < mn/(m+n+1)."""
    return (m * n) / (m + n + 1)


def file_size_kb(path: str) -> float:
    """Actual size on disk (PNG/JPEG have their own compression, so this
    differs from the SVD storage formula -- report both separately)."""
    return os.path.getsize(path) / 1024


# ---------- Module 6: MSE ----------
def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    a = original.astype(np.float64)
    b = reconstructed.astype(np.float64)
    return float(np.mean((a - b) ** 2))


# ---------- Module 7: PSNR ----------
def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")  # identical images
    return float(10 * np.log10((max_val ** 2) / err))

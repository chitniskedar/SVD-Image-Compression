"""evaluate.py  --  Person 2's driver script.

Works with Person 1's modules (image_utils.py, svd_compression.py, main.py).

Workflow (run everything from inside the src/ folder, like main.py expects):
    cd src
    python main.py        # Person 1: creates ../output/compressed_k*.png
    python evaluate.py    # Person 2: metrics, table, graphs -> ../graphs/

evaluate.py loads the original image with Person 1's load_image() and
compares it with the saved PNGs. If a PNG is missing, it rebuilds that
reconstruction using Person 1's apply_svd()/reconstruct_image() and the same
clip + uint8 conversion as save_image(), so results are identical.
"""
import argparse
import csv
import os
import numpy as np
from PIL import Image

from image_utils import load_image                      # Person 1
from svd_compression import apply_svd, reconstruct_image  # Person 1
from metrics import (compressed_size, compression_ratio, storage_saved_percent,
                     break_even_rank, file_size_kb, mse, psnr)
from visualization import (plot_compression_ratio, plot_mse, plot_psnr,
                           plot_comparison)


def load_saved(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def rebuild_like_save_image(U, S, VT, k):
    """Same conversion as image_utils.save_image (clip, then astype uint8)."""
    R = np.clip(reconstruct_image(U, S, VT, k), 0, 255)
    return R.astype(np.uint8).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../input/sample.jpg")
    ap.add_argument("--output_dir", default="../output")
    ap.add_argument("--graphs_dir", default="../graphs")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 20, 50, 100])
    args = ap.parse_args()

    _, A = load_image(args.input)
    m, n = A.shape
    print(f"Image size: {m} x {n}  ({m * n} values, file "
          f"{file_size_kb(args.input):.1f} KB)")
    print(f"SVD storage beats the raw matrix only while "
          f"k < {break_even_rank(m, n):.1f}\n")

    svd_cache = None  # computed only if a PNG is missing
    rows, recons, used_ks = [], [], []
    for k in args.ks:
        if k > min(m, n):
            print(f"[skip] k={k} exceeds the number of singular values")
            continue
        path = os.path.join(args.output_dir, f"compressed_k{k}.png")
        if os.path.exists(path):
            R, disk_kb = load_saved(path), file_size_kb(path)
        else:
            print(f"[note] {path} not found -> rebuilding k={k} with Person 1's functions")
            if svd_cache is None:
                svd_cache = apply_svd(A)
            R, disk_kb = rebuild_like_save_image(*svd_cache, k), float("nan")
        used_ks.append(k)
        recons.append(R)
        rows.append({
            "k": k,
            "svd_values_stored": compressed_size(m, n, k),
            "compression_ratio": round(compression_ratio(m, n, k), 3),
            "storage_saved_%": round(storage_saved_percent(m, n, k), 2),
            "MSE": round(mse(A, R), 3),
            "PSNR_dB": round(psnr(A, R), 3),
            "png_file_KB": round(disk_kb, 1),
        })

    header = list(rows[0].keys())
    print("  ".join(f"{h:>18}" for h in header))
    for r in rows:
        print("  ".join(f"{str(r[h]):>18}" for h in header))

    os.makedirs(args.graphs_dir, exist_ok=True)
    with open(os.path.join(args.graphs_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    plot_compression_ratio(used_ks, [r["compression_ratio"] for r in rows], args.graphs_dir)
    plot_mse(used_ks, [r["MSE"] for r in rows], args.graphs_dir)
    plot_psnr(used_ks, [r["PSNR_dB"] for r in rows], args.graphs_dir)
    plot_comparison(A, recons, used_ks, [r["PSNR_dB"] for r in rows], args.graphs_dir)
    print(f"\nSaved results.csv and 4 figures in '{args.graphs_dir}/'")


if __name__ == "__main__":
    main()

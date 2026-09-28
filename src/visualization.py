"""visualization.py  --  Person 2 (Modules 8, 9)

Three graphs (k vs compression ratio / MSE / PSNR) and a side-by-side
visual comparison of the original and every reconstruction.
"""
import os
import matplotlib
matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt


def _line_plot(ks, values, title, ylabel, path, color):
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, values, marker="o", linewidth=2, color=color)
    for k, v in zip(ks, values):
        plt.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    plt.title(title)
    plt.xlabel("k (number of singular values kept)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ---------- Module 8: Graphs ----------
def plot_compression_ratio(ks, ratios, out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)
    _line_plot(ks, ratios, "k vs Compression Ratio", "Compression Ratio  mn / k(m+n+1)",
               os.path.join(out_dir, "compression_ratio.png"), "tab:blue")


def plot_mse(ks, mses, out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)
    _line_plot(ks, mses, "k vs MSE", "Mean Squared Error",
               os.path.join(out_dir, "mse.png"), "tab:red")


def plot_psnr(ks, psnrs, out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)
    _line_plot(ks, psnrs, "k vs PSNR", "PSNR (dB)",
               os.path.join(out_dir, "psnr.png"), "tab:green")


# ---------- Module 9: Visual comparison ----------
def plot_comparison(original, reconstructions, ks, psnrs, out_dir="graphs"):
    """original: 2-D array; reconstructions: list of 2-D arrays (same order as ks)."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(ks) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    for ax, img, k, p in zip(axes[1:], reconstructions, ks, psnrs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"k = {k}\nPSNR = {p:.1f} dB")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "visual_comparison.png"), dpi=200)
    plt.close()

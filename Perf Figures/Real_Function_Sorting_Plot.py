import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# =========================
# Global paper settings
# =========================
LABEL_FS = 26
TICK_FS  = 26
LEGEND_FS = 20

plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "axes.linewidth": 1.0,
})

INT_BYTES = 4
BYTES_PER_MB = 1024 * 1024

def parse_real_function_sorting(path: str):
    """
    Parses Real_Function_Sorting.txt into:
      dims: sorted list of dimensions
      raw_counts[dim]["I-tree"|"Fs-Tree"] = num (int)
    Expected block format:
      #dimension = X
      I-tree
      Storage = <num>
      Fs-Tree
      Storage = <num>
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = re.split(r"#dimension\s*=\s*", text)
    raw_counts = {}

    for blk in parts[1:]:
        m = re.match(r"(\d+)", blk.strip())
        if not m:
            continue
        dim = int(m.group(1))
        raw_counts.setdefault(dim, {})

        mi = re.search(r"I-tree\s*[\r\n]+Storage\s*=\s*(\d+)", blk)
        mf = re.search(r"Fs-Tree\s*[\r\n]+Storage\s*=\s*(\d+)", blk)

        if mi:
            raw_counts[dim]["I-tree"] = int(mi.group(1))
        if mf:
            raw_counts[dim]["Fs-Tree"] = int(mf.group(1))

    dims = sorted(raw_counts.keys())
    return dims, raw_counts

def itree_storage_mb(num: int) -> float:
    # I-tree: num * 100 * size(int)
    return (num * 100 * INT_BYTES) / BYTES_PER_MB

def fstree_storage_mb(num: int) -> float:
    # Fs-tree: num * size(int)
    return (num * INT_BYTES) / BYTES_PER_MB

def set_log10_y(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))

def main():
    infile = "Real_Function_Sorting.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    dims, raw = parse_real_function_sorting(infile)

    it_mb = []
    fs_mb = []
    for d in dims:
        it_num = raw[d].get("I-tree")
        fs_num = raw[d].get("Fs-Tree")

        if it_num is None or fs_num is None:
            raise ValueError(f"Missing Storage for dim={d}. Found: {raw[d]}")

        it_mb.append(itree_storage_mb(it_num))
        fs_mb.append(fstree_storage_mb(fs_num))

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    ax.bar(x - width/2, it_mb, width=width, label="I-tree")
    ax.bar(x + width/2, fs_mb, width=width, label="Fs-tree")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Storage (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])

    # Log base 10 for storage
    set_log10_y(ax)

    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "fig_real_storage_bar_dimension_vs_storage_log10_mb.pdf")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print("Computed storage (MB):")
    for d, a, b in zip(dims, it_mb, fs_mb):
        print(f"  dim={d}: I-tree={a:.6g} MB, Fs-tree={b:.6g} MB")
    print(f"\nSaved: {outpath}")

if __name__ == "__main__":
    main()
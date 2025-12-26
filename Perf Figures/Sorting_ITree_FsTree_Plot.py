import os
import re
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# =========================
# Global paper settings
# =========================
LABEL_FS = 24
TICK_FS  = 24
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

def parse_sorting_file(path: str):
    """
    Parses Sorting_ITree_FsTree.txt into:
      data[method][dim] = list of (n_functions, time_sec, size_ints)

    Notes:
    - File header says "Intersections" but user defines Col1 as #functions.
      We treat Col1 as x = #functions.
    - Rows may contain 'None' for missing values; those rows are skipped.
    """
    data = {"ITree": {}, "FsTree": {}}
    method = None
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    # examples:
    # 50  0.000197  507
    # 100 0.001145  2423
    row_pat = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s == "ITree":
                method = "ITree"
                dim = None
                continue
            if s == "FsTree":
                method = "FsTree"
                dim = None
                continue

            m_dim = dim_pat.match(s)
            if m_dim and method is not None:
                dim = int(m_dim.group(1))
                data[method].setdefault(dim, [])
                continue

            m_row = row_pat.match(s)
            if m_row and method is not None and dim is not None:
                n = int(m_row.group(1))
                t_str = m_row.group(2)
                sz_str = m_row.group(3)

                if t_str.lower() == "none" or sz_str.lower() == "none":
                    continue

                try:
                    t = float(t_str)
                    sz = int(sz_str)
                except ValueError:
                    continue

                data[method][dim].append((n, t, sz))

    # sort by x
    for meth in data:
        for d in data[meth]:
            data[meth][d].sort(key=lambda x: x[0])
    return data

def set_log10(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))

def compute_storage_mb(method: str, n_funcs: int, size_ints: int) -> float:
    """
    User-specified storage:
      I-tree: size * #functions * int -> MB
      Fs-tree: size * int -> MB
    """
    if method == "ITree":
        bytes_used = size_ints * n_funcs * INT_BYTES
    else:  # FsTree
        bytes_used = size_ints * INT_BYTES
    return bytes_used / BYTES_PER_MB

def plot_time_vs_functions(data, outpath: str):
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    # order in legend
    series = [
        ("ITree", 2, "I-tree(2d)"),
        ("ITree", 3, "I-tree(3d)"),
        ("ITree", 4, "I-tree(4d)"),
        ("FsTree", 2, "FS-tree(2d)"),
        ("FsTree", 3, "FS-tree(3d)"),
        ("FsTree", 4, "FS-tree(4d)"),
    ]

    for method, dim, label in series:
        pts = data.get(method, {}).get(dim, [])
        if not pts:
            continue
        xs = [n for (n, _, _) in pts]
        ys = [t if t > 0 else 1e-12 for (_, t, _) in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    ax.set_xlabel("#Functions")
    ax.set_ylabel("Time (seconds)")
    set_log10(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_storage_vs_functions(data, outpath: str):
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    series = [
        ("ITree", 2, "I-tree(2d)"),
        ("ITree", 3, "I-tree(3d)"),
        ("ITree", 4, "I-tree(4d)"),
        ("FsTree", 2, "FS-tree(2d)"),
        ("FsTree", 3, "FS-tree(3d)"),
        ("FsTree", 4, "FS-tree(4d)"),
    ]

    for method, dim, label in series:
        pts = data.get(method, {}).get(dim, [])
        if not pts:
            continue
        xs = [n for (n, _, _) in pts]
        ys = []
        for (n, _, sz) in pts:
            mb = compute_storage_mb(method, n, sz)
            ys.append(mb if mb > 0 else 1e-12)
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    ax.set_xlabel("#Functions")
    ax.set_ylabel("Storage (MB)")
    set_log10(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def main():
    infile = "Sorting_ITree_FsTree.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or set `infile` to the correct path."
        )

    data = parse_sorting_file(infile)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)

    plot_time_vs_functions(
        data,
        os.path.join(outdir, "fig_sorting_time_vs_functions_log10_seconds.pdf")
    )

    plot_storage_vs_functions(
        data,
        os.path.join(outdir, "fig_sorting_storage_vs_functions_log10_mb.pdf")
    )

    print(f"Saved figures to: {outdir}/")

if __name__ == "__main__":
    main()
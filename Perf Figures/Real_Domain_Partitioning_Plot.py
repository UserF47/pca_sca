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

def parse_real_domain_partitioning(path: str):
    """
    Parses Real_Domain_Partitioning.txt into:
      intersections: sorted list of ints
      times: dict method -> list of times aligned with intersections
    Expected methods: Simplex, Basic, IPA
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # find all blocks starting with "#intersection = X"
    blocks = re.split(r"\#intersection\s*=\s*", text)
    times_map = {"Simplex": {}, "Basic": {}, "IPA": {}}

    for blk in blocks[1:]:
        # blk begins with "100\nSimplex\nTime = ...\n\nBasic\nTime = ...\n..."
        m = re.match(r"(\d+)", blk.strip())
        if not m:
            continue
        inter = int(m.group(1))

        # For each method, capture "Method ... Time = number"
        for method in ["Simplex", "Basic", "IPA"]:
            mm = re.search(rf"\b{method}\b\s*[\r\n]+Time\s*=\s*([0-9]*\.?[0-9]+)", blk)
            if mm:
                times_map[method][inter] = float(mm.group(1))

    intersections = sorted(set().union(*[set(d.keys()) for d in times_map.values()]))

    times = {}
    for method in ["Simplex", "Basic", "IPA"]:
        times[method] = [times_map[method].get(i, np.nan) for i in intersections]

    return intersections, times

def set_log10_seconds(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))

def main():
    infile = "Real_Domain_Partitioning.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    intersections, times = parse_real_domain_partitioning(infile)

    methods = ["Simplex", "Basic", "IPA"]
    x = np.arange(len(intersections))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    for k, method in enumerate(methods):
        ax.bar(x + (k - 1) * width, times[method], width=width, label=method)

    ax.set_xlabel("#Intersections")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in intersections])

    # Recommended: log scale (Simplex is orders of magnitude larger).
    # Comment out the next line if you want linear y-axis.
    set_log10_seconds(ax)

    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "fig_real_domain_bar_time_vs_intersections.pdf")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
import os
import re
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

TARGET_TIMES = [10, 20, 30, 40, 50]

def annotate_points(ax, xs, ys, fontsize=16, skip_x=None):
    """
    Annotate data points on a log-scale y-axis.
    """
    if skip_x is None:
        skip_x = set()
    for x, y in zip(xs, ys):
        if x in skip_x:
            continue
        ax.annotate(
            f"{y}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),   # vertical offset (works well for log scale)
            ha="center",
            fontsize=fontsize,
            clip_on=True
        )

def parse_scale_file(path: str):
    """
    Expected format:
      Simplex
      dim=3
      10   1234
      20   5678
      ...
      dim=4
      ...

      Vertex
      dim=3
      ...

    Rows are: time(min) <whitespace> intersections
    Returns:
      data[method][dim] = dict(time_min -> intersections)
    """
    data = {"Simplex": {}, "Vertex": {}}
    method = None
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    row_pat = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s == "Simplex":
                method = "Simplex"
                dim = None
                continue
            if s == "Vertex":
                method = "Vertex"
                dim = None
                continue

            m_dim = dim_pat.match(s)
            if m_dim and method is not None:
                dim = int(m_dim.group(1))
                data[method].setdefault(dim, {})
                continue

            m_row = row_pat.match(s)
            if m_row and method is not None and dim is not None:
                t = int(m_row.group(1))
                inter = int(m_row.group(2))
                data[method][dim][t] = inter

    return data

def main():
    infile = "FC_Simplex_Vertex_Scale.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    data = parse_scale_file(infile)

    # Correct legend capitalization + dimensions
    LABEL_MAP = {
        ("Simplex", 3): "Simplex(3d)",
        ("Simplex", 4): "Simplex(4d)",
        ("Vertex",  3): "Vertex(3d)",
        ("Vertex",  4): "Vertex(4d)",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    all_ys = []

    for (method, dim), label in LABEL_MAP.items():
        time_to_inter = data.get(method, {}).get(dim, {})
        xs, ys = [], []

        for t in TARGET_TIMES:
            if t in time_to_inter:
                xs.append(t)                 # x = time (minutes)
                ys.append(time_to_inter[t])  # y = intersections
                all_ys.append(time_to_inter[t])

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
            if method == "Vertex" and dim == 3:
                annotate_points(ax, xs, ys, skip_x={50})
            else:
                annotate_points(ax, xs, ys)
    # Axes
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("#Intersections")
    ax.set_xticks(TARGET_TIMES)

    # Log-scale with 10^k formatting
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9))
    )

    y_max = max(all_ys) if all_ys else 100
    ax.set_ylim(bottom=100, top=y_max * 1.35)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "fig_scalability_intersections_vs_time_log10.pdf")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
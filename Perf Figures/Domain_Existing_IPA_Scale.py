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

def parse_scale_file(path: str):
    """
    Parses Domain_Existing_IPA_Scale.txt into:
      data[method][dim] = dict(time_min -> intersections)

    Expected blocks:
      Baseline
      dim=3
      10   26902
      ...
      IPA
      dim=4
      ...
    """
    data = {"Baseline": {}, "IPA": {}}
    method = None
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    row_pat = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s == "Baseline":
                method = "Baseline"
                dim = None
                continue
            if s == "IPA":
                method = "IPA"
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

def annotate_points(ax, xs, ys, fontsize=16):
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=fontsize,
            clip_on=True
        )

def main():
    infile = "Domain_Existing_IPA_Scale.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    data = parse_scale_file(infile)

    LABEL_MAP = {
        ("Baseline", 3): "Basic (3D)",
        ("Baseline", 4): "Basic (4D)",
        ("IPA",      3): "IPA (3D)",
        ("IPA",      4): "IPA (4D)",
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
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
            annotate_points(ax, xs, ys)
            all_ys.extend(ys)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("#Intersections")
    ax.set_xticks(TARGET_TIMES)

    # log10 with 10^k labels
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    if all_ys:
        ax.set_ylim(top=max(all_ys) * 1.3)

    # Optional minor ticks (visual guidance)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))

    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "fig_domain_scale_intersections_vs_time_log10.pdf")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
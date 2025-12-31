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

# Label mapping for plot legends
LABEL_MAP = {
    "Baseline": "Basic",
    "IPA": "IPA",
}

def parse_existing_ipa(path: str):
    """
    Parses Domain_Existing_IPA.txt (see provided format) into:
      data[method][dim] = list of (intersections, time_seconds), sorted by intersections
    Skips empty lines.
    """
    data = { "Baseline": {}, "IPA": {}}
    method = None
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    row_pat = re.compile(r"^\s*(\d+)\s+([0-9]*\.?[0-9]+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s ==  "Baseline":
                method =  "Baseline"
                dim = None
                continue
            if s == "IPA":
                method = "IPA"
                dim = None
                continue

            m_dim = dim_pat.match(s)
            if m_dim and method is not None:
                dim = int(m_dim.group(1))
                data[method].setdefault(dim, [])
                continue

            m_row = row_pat.match(s)
            if m_row and method is not None and dim is not None:
                inter = int(m_row.group(1))
                t = float(m_row.group(2))
                data[method][dim].append((inter, t))

    for meth in data:
        for d in data[meth]:
            data[meth][d].sort(key=lambda x: x[0])

    return data

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def set_log10_seconds(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

def plot_time_vs_dimension_at_intersections(data, target_intersections: int, outpath: str):
    """
    Figure 1:
      x-axis: dimension (at fixed #intersections=target_intersections)
      y-axis: time (seconds), log10
      Compare Existing vs IPA
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    dims = sorted(set(data[ "Baseline"].keys()) | set(data["IPA"].keys()))

    def time_at(method: str, dim: int, inter: int):
        for i, t in data.get(method, {}).get(dim, []):
            if i == inter:
                return t
        return None

    for method in [ "Baseline", "IPA"]:
        xs, ys = [], []
        for d in dims:
            t = time_at(method, d, target_intersections)
            if t is None:
                continue
            xs.append(d)
            ys.append(t if t > 0 else 1e-12)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=LABEL_MAP.get(method, method))

    ax.set_xlabel("Dimension(#intersection=2500)")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(dims)

    set_log10_seconds(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_time_vs_intersections_at_dim(data, target_dim: int, outpath: str):
    """
    Figure 2:
      x-axis: #intersections (at fixed dimension=target_dim)
      y-axis: time (seconds), log10
      Compare Existing vs IPA
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    for method in [ "Baseline", "IPA"]:
        pts = data.get(method, {}).get(target_dim, [])
        if not pts:
            continue
        xs = [i for i, _ in pts]
        ys = [t if t > 0 else 1e-12 for _, t in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, label=LABEL_MAP.get(method, method))

    ax.set_xlabel("#Intersections(dim=4)")
    ax.set_ylabel("Time (seconds)")

    set_log10_seconds(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def main():
    infile = "Domain_Existing_IPA.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    data = parse_existing_ipa(infile)

    outdir = "paper_figures"
    ensure_dir(outdir)

    # Figure 1: x = dimension, fixed intersections=2000
    plot_time_vs_dimension_at_intersections(
        data,
        target_intersections=2500,
        outpath=os.path.join(outdir, "fig_domain_time_vs_dimension_inter2000_log10_seconds.pdf")
    )

    # Figure 2: x = intersections, fixed dimension=3
    plot_time_vs_intersections_at_dim(
        data,
        target_dim=4,
        outpath=os.path.join(outdir, "fig_domain_time_vs_intersections_dim3_log10_seconds.pdf")
    )

    print(f"Saved figures to: {outdir}/")

if __name__ == "__main__":
    main()
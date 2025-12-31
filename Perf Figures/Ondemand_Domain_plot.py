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

def set_log10_seconds(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(2,3,4,5,6,7,8,9))
    )

def parse_ondemand_domain(path: str):
    """
    Returns:
      data[dim] = list of (n_funcs, time1, time2)
    """
    data = {}
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    row_pat = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or set(s) == {"="}:
                continue

            m_dim = dim_pat.match(s)
            if m_dim:
                dim = int(m_dim.group(1))
                data.setdefault(dim, [])
                continue

            m_row = row_pat.match(s)
            if m_row and dim is not None:
                n = int(m_row.group(1))
                t1 = float(m_row.group(2))
                t2 = float(m_row.group(3))
                data[dim].append((n, t1, t2))

    for d in data:
        data[d].sort(key=lambda x: x[0])

    return data

# -------------------------
# Figure 1
# -------------------------
def plot_time_vs_functions_dim(data, dim: int, outpath: str):
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    pts = data.get(dim, [])
    if pts:
        xs = [n for (n, _, _) in pts]
        y1 = [t if t > 0 else 1e-12 for (_, t, _) in pts]
        y2 = [t if t > 0 else 1e-12 for (_, _, t) in pts]

        ax.plot(xs, y1, marker="o", linewidth=2,
                label="On-demand(first attempt)")
        ax.plot(xs, y2, marker="o", linewidth=2,
                label="On-demand(second attempt)")

    ax.set_xlabel("#Functions")
    ax.set_ylabel("Time (seconds)")
    set_log10_seconds(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# -------------------------
# Figure 2
# -------------------------
def plot_time_vs_dimension_at_funcs(data, target_funcs: int, outpath: str):
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    dims = sorted(data.keys())
    xs, y1s, y2s = [], [], []

    for d in dims:
        for (n, t1, t2) in data[d]:
            if n == target_funcs:
                xs.append(d)
                y1s.append(t1 if t1 > 0 else 1e-12)
                y2s.append(t2 if t2 > 0 else 1e-12)
                break

    if xs:
        ax.plot(xs, y1s, marker="o", linewidth=2,
                label="On-demand(first attempt)")
        ax.plot(xs, y2s, marker="o", linewidth=2,
                label="On-demand(second attempt)")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(xs)
    set_log10_seconds(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def main():
    infile = "Ondemand_Domain.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Cannot find {infile}")

    data = parse_ondemand_domain(infile)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)

    # Your defaults
    plot_time_vs_functions_dim(
        data, dim=4,
        outpath=os.path.join(outdir,
                             "fig_ondemand_time_vs_functions_dim4_log10.pdf")
    )

    plot_time_vs_dimension_at_funcs(
        data, target_funcs=300,
        outpath=os.path.join(outdir,
                             "fig_ondemand_time_vs_dimension_funcs300_log10.pdf")
    )

    print(f"Saved figures to: {outdir}/")

if __name__ == "__main__":
    main()
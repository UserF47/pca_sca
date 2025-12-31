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
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))

def parse_on_demand_point(path: str):
    """
    Parses OnDemand_Point.txt into:
      ondemand[dim] = list of (n_funcs, time1, time2)
      linear[dim]   = list of (n_funcs, time1)

    File structure (as provided):
      On-demand
      dim=6
      2000  t1  t2
      ...
      Linear Sorting
      dim=6
      2000  t1
      ...
    """
    ondemand = {}
    linear = {}

    mode = None  # "On-demand" or "Linear Sorting"
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    row3_pat = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$")
    row2_pat = re.compile(r"^\s*(\d+)\s+(\S+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or set(s) == {"="}:
                continue

            if s == "On-demand":
                mode = "On-demand"
                dim = None
                continue
            if s == "Linear Sorting":
                mode = "Linear Sorting"
                dim = None
                continue

            m_dim = dim_pat.match(s)
            if m_dim and mode is not None:
                dim = int(m_dim.group(1))
                if mode == "On-demand":
                    ondemand.setdefault(dim, [])
                else:
                    linear.setdefault(dim, [])
                continue

            if mode == "On-demand":
                m_row = row3_pat.match(s)
                if m_row and dim is not None:
                    n = int(m_row.group(1))
                    t1 = float(m_row.group(2))
                    t2 = float(m_row.group(3))
                    ondemand[dim].append((n, t1, t2))
            elif mode == "Linear Sorting":
                m_row = row2_pat.match(s)
                if m_row and dim is not None:
                    n = int(m_row.group(1))
                    t1 = float(m_row.group(2))
                    linear[dim].append((n, t1))

    for d in ondemand:
        ondemand[d].sort(key=lambda x: x[0])
    for d in linear:
        linear[d].sort(key=lambda x: x[0])

    return ondemand, linear

def figure1_time_vs_functions(ondemand, linear, default_dim: int, outpath: str):
    """
    Figure 1:
      x: #functions (default dimension = default_dim)
      y: time (seconds), log10
      legend:
        on-demand(time1), on-demand(time2), linear sorting(time1)
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    od_pts = ondemand.get(default_dim, [])
    ls_pts = linear.get(default_dim, [])

    if od_pts:
        xs = [n for (n, _, _) in od_pts]
        y1 = [t if t > 0 else 1e-12 for (_, t, _) in od_pts]
        y2 = [t if t > 0 else 1e-12 for (_, _, t) in od_pts]
        ax.plot(xs, y1, marker="o", linewidth=2, label="On-demand(first attempt)")
        ax.plot(xs, y2, marker="o", linewidth=2, label="On-demand(second attempt)")

    if ls_pts:
        xs = [n for (n, _) in ls_pts]
        ys = [t if t > 0 else 1e-12 for (_, t) in ls_pts]
        ax.plot(xs, ys, marker="o", linewidth=2, label="Linear Sorting")

    ax.set_xlabel("#Functions")
    ax.set_ylabel("Time (seconds)")
    set_log10_seconds(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def figure2_time_vs_dimension(ondemand, linear, default_funcs: int, outpath: str):
    """
    Figure 2:
      x: #dimension (default #functions = default_funcs)
      y: time (seconds), log10
      legend:
        on-demand(time1), on-demand(time2), linear sorting(time1)
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    dims = sorted(set(ondemand.keys()) | set(linear.keys()))

    # On-demand(time1/time2) at fixed #functions
    od_x, od_y1, od_y2 = [], [], []
    for d in dims:
        pts = ondemand.get(d, [])
        for (n, t1, t2) in pts:
            if n == default_funcs:
                od_x.append(d)
                od_y1.append(t1 if t1 > 0 else 1e-12)
                od_y2.append(t2 if t2 > 0 else 1e-12)
                break

    if od_x:
        ax.plot(od_x, od_y1, marker="o", linewidth=2, label="On-demand(first attempt)")
        ax.plot(od_x, od_y2, marker="o", linewidth=2, label="On-demand(second attempt)")

    # Linear Sorting(time1) at fixed #functions
    ls_x, ls_y = [], []
    for d in dims:
        pts = linear.get(d, [])
        for (n, t1) in pts:
            if n == default_funcs:
                ls_x.append(d)
                ls_y.append(t1 if t1 > 0 else 1e-12)
                break

    if ls_x:
        ax.plot(ls_x, ls_y, marker="o", linewidth=2, label="Linear Sorting")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(dims)
    set_log10_seconds(ax)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def main():
    infile = "OnDemand_Point.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    ondemand, linear = parse_on_demand_point(infile)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)

    # Your defaults
    default_dim = 6
    default_funcs = 6000

    figure1_time_vs_functions(
        ondemand, linear, default_dim=default_dim,
        outpath=os.path.join(outdir, "fig_ondemand_vs_linear_time_vs_functions_dim6_log10.pdf")
    )

    figure2_time_vs_dimension(
        ondemand, linear, default_funcs=default_funcs,
        outpath=os.path.join(outdir, "fig_ondemand_vs_linear_time_vs_dimension_funcs6000_log10.pdf")
    )

    print(f"Saved figures to: {outdir}/")

if __name__ == "__main__":
    main()
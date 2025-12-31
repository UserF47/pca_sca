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

# What you want on x-axis
DIMS = [4, 6, 8, 10]

# Choose the fixed #functions for the bar chart
TARGET_FUNCS = 6000  # change to 300 / 2000 / etc if needed

def parse_on_demand_point(path: str):
    """
    Parses OnDemand_Point.txt into:
      ondemand[dim] = list of (n_funcs, time1, time2)
      linear[dim]   = list of (n_funcs, time1)

    Expected sections:
      On-demand
      dim=...
      n  t1  t2
      ...
      Linear Sorting
      dim=...
      n  t1
      ...
    """
    ondemand = {}
    linear = {}

    mode = None
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
                m = row3_pat.match(s)
                if m and dim is not None:
                    n = int(m.group(1))
                    t1 = float(m.group(2))
                    t2 = float(m.group(3))
                    ondemand[dim].append((n, t1, t2))

            elif mode == "Linear Sorting":
                m = row2_pat.match(s)
                if m and dim is not None:
                    n = int(m.group(1))
                    t1 = float(m.group(2))
                    linear[dim].append((n, t1))

    # sort by #functions
    for d in ondemand:
        ondemand[d].sort(key=lambda x: x[0])
    for d in linear:
        linear[d].sort(key=lambda x: x[0])

    return ondemand, linear

def time_at_n_on_demand(ondemand, dim: int, n: int):
    for (nn, t1, t2) in ondemand.get(dim, []):
        if nn == n:
            return t1, t2
    return None, None

def time_at_n_linear(linear, dim: int, n: int):
    for (nn, t1) in linear.get(dim, []):
        if nn == n:
            return t1
    return None

def set_log10_y(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))

def main():
    infile = "OnDemand_Point.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file."
        )

    ondemand, linear = parse_on_demand_point(infile)

    # Collect bar heights in the requested dim order
    od_first = []
    od_second = []
    lin_sort = []

    missing = []
    for d in DIMS:
        t1, t2 = time_at_n_on_demand(ondemand, d, TARGET_FUNCS)
        tl = time_at_n_linear(linear, d, TARGET_FUNCS)

        if t1 is None or t2 is None or tl is None:
            missing.append(d)

        # Use NaN for missing -> bar won't render (better than plotting fake 0)
        od_first.append(t1 if t1 is not None else np.nan)
        od_second.append(t2 if t2 is not None else np.nan)
        lin_sort.append(tl if tl is not None else np.nan)

    if missing:
        print(f"[Warning] Missing #functions={TARGET_FUNCS} for dimensions: {missing}")
        print("Bars for missing entries will be empty (NaN).")

    x = np.arange(len(DIMS))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    ax.bar(x - width, od_first,  width=width, label="On-demand:first")
    ax.bar(x,         od_second, width=width, label="On-demand:second")
    ax.bar(x + width, lin_sort,  width=width, label="Linear sorting")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DIMS])

    set_log10_y(ax)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"fig_bar_time_vs_dimension_funcs{TARGET_FUNCS}_log10.pdf")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
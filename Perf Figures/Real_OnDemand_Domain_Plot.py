import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

TARGET_DIMS = [3, 4, 5]
TARGET_FUNCS = [100, 300, 500]

HATCH = {
    3: "///",
    4: "\\\\",
    5: "xx",
}

def parse_ondemand_domain(path: str):
    """
    data[dim][#functions] = (time1, time2)
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
                data.setdefault(dim, {})
                continue

            m_row = row_pat.match(s)
            if m_row and dim is not None:
                n = int(m_row.group(1))
                t1 = float(m_row.group(2))
                t2 = float(m_row.group(3))
                data[dim][n] = (t1, t2)

    return data

def set_log10_y(ax):
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(2,3,4,5,6,7,8,9))
    )

def main():
    infile = "Ondemand_Domain.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Cannot find {infile}")

    data = parse_ondemand_domain(infile)

    x = np.arange(len(TARGET_FUNCS))
    group_width = 0.84
    n_bars_per_group = len(TARGET_DIMS) * 2
    bar_w = group_width / n_bars_per_group

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    attempt_style = {
        "first":  {"alpha": 1.0, "linewidth": 0.0},
        "second": {"alpha": 0.75, "linewidth": 1.5},
    }

    bar_idx = 0
    for d in TARGET_DIMS:
        for attempt in ["first", "second"]:
            offset = (bar_idx - (n_bars_per_group - 1) / 2) * bar_w
            ys = []

            for n in TARGET_FUNCS:
                if d in data and n in data[d]:
                    t = data[d][n][0] if attempt == "first" else data[d][n][1]
                    ys.append(t)
                else:
                    ys.append(np.nan)

            ax.bar(
                x + offset,
                ys,
                width=bar_w,
                hatch=HATCH[d],
                alpha=attempt_style[attempt]["alpha"],
                edgecolor="black" if attempt == "second" else None,
                linewidth=attempt_style[attempt]["linewidth"],
                )
            bar_idx += 1

    ax.set_xlabel("#Functions")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in TARGET_FUNCS])

    # --- log10 y-axis ---
    set_log10_y(ax)

    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.35)

    # -------- Legends (no group titles) --------
    legend_attempt = [
        Patch(facecolor="gray", edgecolor="none", label="First attempt"),
        Patch(facecolor="gray", edgecolor="black", linewidth=1.5, alpha=0.75,
              label="Second attempt"),
    ]
    leg1 = ax.legend(handles=legend_attempt, loc="upper left", frameon=True)

    legend_dim = [
        Patch(facecolor="white", edgecolor="black", hatch=HATCH[3], label="Dim=3"),
        Patch(facecolor="white", edgecolor="black", hatch=HATCH[4], label="Dim=4"),
        Patch(facecolor="white", edgecolor="black", hatch=HATCH[5], label="Dim=5"),
    ]
    leg2 = ax.legend(handles=legend_dim, loc="upper right", frameon=True)

    ax.add_artist(leg1)

    outdir = "paper_figures"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(
        outdir,
        "fig_ondemand_bar_funcs100_300_500_dims3_4_5_log10.pdf"
    )
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
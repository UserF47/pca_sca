import os
import re
import math
import matplotlib.pyplot as plt

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

def seconds_to_minutes(v: float) -> float:
    return v / 60.0

def parse_fc_file(path: str):
    """
    Parses FC_Simplex_Vertex.txt into:
      data[method][dim] = list of (intersections, time_seconds)
    Skips 'None' entries.
    """
    data = {"Simplex": {}, "Vertex": {}}
    method = None
    dim = None

    dim_pat = re.compile(r"^\s*dim\s*=\s*(\d+)\s*$")
    row_pat = re.compile(r"^\s*(\d+)\s+(\S+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.lower() == "simplex":
                method = "Simplex"
                dim = None
                continue
            if s.lower() == "vertex":
                method = "Vertex"
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
                t_str = m_row.group(2)
                if t_str.lower() == "none":
                    continue
                try:
                    t = float(t_str)
                except ValueError:
                    continue
                data[method][dim].append((inter, t))

    # Sort by intersections
    for meth in data:
        for d in data[meth]:
            data[meth][d].sort(key=lambda x: x[0])
    return data

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def plot_fig1_time_vs_intersections_dim3(data, outpath: str):
    """
    Figure 1: feasibility checking time vs #intersections (dim=3)
    - y: minutes (log scale)
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    dim = 3
    for method in ["Simplex", "Vertex"]:
        pts = data.get(method, {}).get(dim, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [seconds_to_minutes(p[1]) for p in pts]

        # log-scale can't plot non-positive values
        ys = [y if y > 0 else 1e-12 for y in ys]

        ax.plot(xs, ys, marker="o", linewidth=2, label=method)

    ax.set_xlabel("#Intersections(dim=3)")
    ax.set_ylabel("Time (min)")
    ax.set_yscale("log")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_fig2_time_vs_dimension_inter300(data, outpath: str):
    """
    Figure 2: feasibility checking time vs #dimension (#intersection=300)
    - y: minutes (log scale)
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    target_intersections = 300
    dims = sorted(set(data.get("Simplex", {}).keys()) | set(data.get("Vertex", {}).keys()))

    def get_time(method: str, dim: int, target_inter: int):
        pts = data.get(method, {}).get(dim, [])
        for inter, t in pts:
            if inter == target_inter:
                return t
        return None

    for method in ["Simplex", "Vertex"]:
        xs = []
        ys = []
        for d in dims:
            t = get_time(method, d, target_intersections)
            if t is None:
                continue
            y = seconds_to_minutes(t)
            ys.append(y if y > 0 else 1e-12)
            xs.append(d)

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=method)

    ax.set_xlabel("Dimension(#intersection=300)")
    ax.set_ylabel("Time (min)")
    ax.set_yscale("log")
    ax.set_xticks(dims)  # show integer dimensions
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def main():
    infile = "FC_Simplex_Vertex.txt"
    if not os.path.exists(infile):
        raise FileNotFoundError(
            f"Cannot find {infile}. Put this script in the same folder as the txt file, "
            f"or change `infile` to the correct path."
        )

    data = parse_fc_file(infile)

    outdir = "paper_figures"
    ensure_dir(outdir)

    plot_fig1_time_vs_intersections_dim3(
        data,
        os.path.join(outdir, "fig1_fc_time_vs_intersections_dim3_log_minutes.pdf")
    )

    plot_fig2_time_vs_dimension_inter300(
        data,
        os.path.join(outdir, "fig2_fc_time_vs_dimension_inter300_log_minutes.pdf")
    )

    print(f"Saved to: {outdir}/")

if __name__ == "__main__":
    main()
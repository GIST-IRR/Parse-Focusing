import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.stats as stats
from csv import reader

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54


def main(filepath, xlabel, ylabel, output, bins=8):
    # Load scatter data
    xs = []
    ys = []
    with open(filepath, "r") as f:
        rd = reader(f)
        for r in rd:
            xs.append(float(r[0]))
            ys.append(float(r[1]))

    # Start with a square Figure.
    fig = plt.figure(figsize=(12 * cm, 12 * cm))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(6, 1),
        height_ratios=(1, 6),
        wspace=0.05,
        hspace=0.05,
    )
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # Correlation Coefficient
    corr = stats.pearsonr(xs, ys)
    ax.text(
        0.07,
        0.15,
        f"$r={corr[0]:.2f}$\n$P={corr[1]:.2f}$",
        transform=ax.transAxes,
        math_fontfamily="dejavuserif",
        fontsize=11,
        verticalalignment="top",
    )
    # Draw the scatter plot and marginals.
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(xs, ys)
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10)

    # now determine nice limits by hand:
    # binwidth = 0.25
    # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # lim = (int(xymax / binwidth) + 1) * binwidth
    # bins = np.arange(-lim, lim + binwidth, binwidth)

    ax_histx.hist(xs, bins=bins)
    ax_histy.hist(ys, bins=bins, orientation="horizontal")
    # Set font size of axis labels
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # Save figure
    plt.savefig(output.with_suffix(".png"), format="png", bbox_inches="tight")
    plt.savefig(output.with_suffix(".svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="raw/scatter_with_hist.csv",
        help="Path to the csv file",
    )
    parser.add_argument(
        "--xlabel", type=str, default="F1", help="Label for the x-axis"
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Negative Log Likelihood",
        help="Label for the y-axis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/f1_ll_scatter_with_hist.png",
        help="Path to the output file",
        type=Path,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Number of bins for the histograms",
    )
    args = parser.parse_args()
    main(args.filepath, args.xlabel, args.ylabel, args.output, args.bins)

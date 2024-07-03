import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.stats as stats
from csv import reader

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54


# some random data
def csv_open(file):
    x = []
    y = []
    with open(file, "r") as f:
        rd = reader(f)
        for r in rd:
            x.append(float(r[0]))
            y.append(float(r[1]))
    return x, y


def main(filepath, labels, xlabel, ylabel, output):
    xs = []
    ys = []
    for file in filepath:
        f1, ll = csv_open(file)
        xs.append(f1)
        ys.append(ll)

    # Start with a square Figure.
    fig = plt.figure(figsize=(12 * cm, 12 * cm))
    # Create the Axes.
    ax = fig.add_subplot()
    # Correlation Coefficient
    corrs = []
    for x, y in zip(xs, ys):
        corrs.append(stats.pearsonr(x, y))

    # Draw the scatter plot and marginals.
    markers = [
        "o",
        "^",
        "s",
        "x",
        "D",
        "P",
        "v",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "8",
        "p",
        "h",
        "H",
        "+",
        "X",
    ]
    for x, y, label, marker in zip(xs, ys, labels, markers):
        ax.scatter(x, y, label=label, marker=marker, s=15)

    ax.grid(alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10)
    ax.legend(fontsize=11, handletextpad=0.0, borderpad=0.6)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # Save the figure.
    output = Path(output)
    plt.savefig(output.with_suffix(".png"), format="png", bbox_inches="tight")
    plt.savefig(output.with_suffix(".svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        nargs="+",
        default="analyzer/raw/ftn_nt4500_f1_ll.csv analyzer/cache/ftn_nt250_f1_ll.csv analyzer/cache/ftn_nt30_f1_ll.csv",
        help="Path to the CSV file containing the F1 and NLL values.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default="NT=4500 NT=250 NT=30",
        help="Labels for the scatter plot.",
    )
    parser.add_argument(
        "--xlabel",
        default="F1",
        help="Label for the x-axis.",
    )
    parser.add_argument(
        "--ylabel",
        default="Negative Log Likelihood (NLL)",
        help="Label for the y-axis.",
    )
    parser.add_argument(
        "--output",
        default="results/f1_ll_scatter_comparison.png",
        help="Path to the output file. The file extension determines the output format.",
    )
    args = parser.parse_args()

    main(
        args.filepath,
        args.labels,
        args.xlabel,
        args.ylabel,
        args.output,
    )
    print("Done!")

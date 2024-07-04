import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54


def main(file_path, output_path, difference=False):
    data = pd.read_csv(file_path)
    data = data.apply(pd.to_numeric, errors="ignore")

    if difference:
        diff = []
        for i, r in data.iterrows():
            diff.append(r["new f1"] - r["org f1"])

    # Group processing
    if "group id" in data.columns:
        gs = data["group id"].value_counts(sort=False)
        group_labels = gs.index
        group_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    x = data["model name"]
    y = data["new f1"]
    yerr = data["err"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    if difference:
        y = diff
    else:
        ymin = 5 * (y.min() // 5)
        ymax = (y.max() + yerr[y.argmax()]) + 1
        ax.set_ylim(ymin, ymax)

    p = ax.bar(x, y, yerr=yerr, capsize=4)
    for i in range(len(data)):
        gid = group_labels.get_loc(data["group id"][i])
        p[i].set_color(group_color[gid])
    ax.set_ylabel("Difference in S-F1", fontsize=13)
    ax.grid(alpha=0.3, axis="y")

    plt.xticks(rotation=45, ha="right")

    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(
        output_path.with_suffix(".svg"), format="svg", bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="The path to the input CSV file",
        default="fig_rule_util/eng_rule_util.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The path to the output image file",
        default="results/homo_hetero_diff_test.png",
    )
    parser.add_argument("--difference", action="store_true")
    args = parser.parse_args()
    main(args.input, args.output, difference=args.difference)

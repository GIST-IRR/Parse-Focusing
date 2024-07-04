import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"


# Single graph: tick size 15, legend size 15, label size 19
# Multi graph: tick size 17, legend size 17, label size 30
def main(
    file_path,
    output_path,
    split_by_group=False,
    n_col=5,
    n_row=2,
    tick_size=15,
    legend_size=15,
    label_size=19,
):
    data = pd.read_csv(file_path)
    data = data.apply(pd.to_numeric, errors="ignore")

    # Group processing
    if "group id" in data.columns:
        gs = data["group id"].value_counts(sort=False)
        n_group = len(gs)
        group_labels = gs.index
        group_lengths = gs.values
        groups = [data[data["group id"] == l] for l in group_labels]
    else:
        groups = [data]

    # Draw line charts for each model.
    if split_by_group:
        x_size = 6 * n_col - 5
        y_size = 6 * n_row - 2
        fig, axes = plt.subplots(
            n_row,
            n_col,
            figsize=(x_size, y_size),
            constrained_layout=True,
        )

        def draw_figure(title, data, ax):
            for _, row in data.iterrows():
                model_name = row["model name"]
                x = data.columns[2:].astype(int)
                y = row[2:].values
                ax.plot(
                    x,
                    y,
                    label=model_name,
                )

            # Set the chart title and axis labels.
            # plt.title("Sentence Length vs Rule Types")
            ax.tick_params(axis="x", labelsize=tick_size)
            ax.tick_params(axis="y", labelsize=tick_size)
            ax.legend(fontsize=legend_size)
            ax.grid(True)
            ax.set_title(title, fontsize=x_size)

        for title, data, ax in zip(group_labels, groups, axes.flatten()):
            draw_figure(title, data, ax)

        fig.supxlabel("Sentence Length", fontsize=label_size)
        fig.supylabel("Number of Rule Types", fontsize=label_size)
    else:
        plt.figure(figsize=(6, 6))

        linestyles = ["solid", "dashed", "dotted", "dashdot"]
        linecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for g, ls in zip(groups, linestyles):
            start_idx = g.index[0]
            for i, row in g.iterrows():
                model_name = row["model name"]
                x = g.columns[2:].astype(int)
                y = row[2:].values
                plt.plot(
                    x,
                    y,
                    label=model_name,
                    linestyle=ls,
                    color=linecolors[i - start_idx],
                )

        # Set the font size
        plt.legend(fontsize=legend_size)
        plt.grid(True)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.xlabel("Sentence Length", fontsize=label_size)
        plt.ylabel("Number of Rule Types", fontsize=label_size)

    output_path = Path(output_path)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(
        output_path.with_suffix(".svg"), format="svg", bbox_inches="tight"
    )


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
        default="eng_rule_util.png",
    )
    parser.add_argument(
        "--split_by_group",
        action="store_true",
        help="Whether to split the data by group",
    )
    parser.add_argument(
        "--n_col",
        type=int,
        default=5,
        help="The number of columns in the output figure",
    )
    parser.add_argument(
        "--n_row",
        type=int,
        default=2,
        help="The number of rows in the output figure",
    )
    parser.add_argument(
        "--tick_size",
        type=int,
        default=15,
        help="The font size of the tick labels",
    )
    parser.add_argument(
        "--legend_size",
        type=int,
        default=15,
        help="The font size of the legend",
    )
    parser.add_argument(
        "--label_size",
        type=int,
        default=19,
        help="The font size of the axis labels",
    )
    args = parser.parse_args()

    main(args.input, args.output, args.split_by_group, args.n_col, args.n_row)

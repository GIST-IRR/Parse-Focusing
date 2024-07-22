from pathlib import Path
import pickle
import argparse
from collections import Counter
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_trees
from torch_support.metric import preprocess_span

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54


def count_common_from_span(spans, length, gold=None):
    # Preprocessing
    spans = [preprocess_span(s, length) for s in spans]
    # Set to list
    spans = [list(s) for s in spans]
    # Count commons for each spans
    common = Counter(chain(*spans))
    if gold is not None:
        # TODO: Not completed
        gold = preprocess_span(gold, length)
        common = {k: v for k, v in common.items() if k in gold}


def main(
    input_path,
    trees,
    output_path,
    n_bins=16,
    tick_font_size=5,
    label_font_size=6,
    svg=True,
):
    # Load data
    with input_path.open("rb") as f:
        cuc_bins = pickle.load(f)

    trees = [load_trees(t, use_span=True) for t in tqdm(trees)]

    cuc_ox = np.zeros((n_bins, 2))
    cuc_o_sets = [set() for _ in range(n_bins)]
    cuc_x_sets = [set() for _ in range(n_bins)]

    for c in cuc_bins:
        for s, v in c.items():
            ox = v[0]
            bins = v[1] - 1
            if ox == 0:
                cuc_ox[bins][0] += 1
                cuc_x_sets[bins].add(s)
            elif ox == 1:
                cuc_ox[bins][1] += 1
                cuc_o_sets[bins].add(s)

    weighted_cuc_ox = cuc_ox * np.arange(1, n_bins + 1)[:, None]

    fig, ax = plt.subplots(figsize=(6, 3))
    bottom = np.zeros(n_bins)
    for i, cuc in enumerate(weighted_cuc_ox.T):
        label = "Not in gold" if i == 0 else "In gold"
        p = ax.bar(np.arange(n_bins) + 1, cuc, label=label, bottom=bottom)
        bottom += cuc
        # ax.bar_label(p, label_type="center")
    ax.legend(fontsize=tick_font_size / cm)
    ax.set_xlabel(
        "Number of parsers sharing common span", fontsize=label_font_size / cm
    )
    ax.set_ylabel("Count of common spans", fontsize=label_font_size / cm)
    plt.xlim(0, n_bins + 1)
    xt = list(range(0, n_bins + 2, 2))
    xt_label = [str(int(x)) for x in xt]
    xt_label[0] = xt_label[0] + "\nSpecific"
    xt_label[-1] = xt_label[-1] + "\nCommon"

    plt.yticks(fontsize=tick_font_size / cm)
    plt.xticks(xt, xt_label, fontsize=tick_font_size / cm)

    plt.savefig(output_path, bbox_inches="tight")
    if svg:
        output_path = Path(output_path)
        output_path = output_path.with_suffix(".svg")
        plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=Path,
        default="analyzer/cache/cuc_bins_FTN4500.pkl",
    )
    parser.add_argument("--trees", nargs="+", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument(
        "--output_path", type=str, default="results/cuc_hist.png"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=16,
        help="Number of bins for histogram. Default: 16.",
    )
    parser.add_argument(
        "--tick_size",
        type=float,
        default=5,
        help="Font size for ticks. Default: 5.",
    )
    parser.add_argument(
        "--label_size",
        type=float,
        default=6,
        help="Font size for labels. Default: 6.",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Save as svg. Default: True.",
        default=True,
    )
    args = parser.parse_args()

    main(
        args.input_path,
        args.trees,
        args.output_path,
        args.n_bins,
        args.tick_size,
        args.label_size,
        args.svg,
    )

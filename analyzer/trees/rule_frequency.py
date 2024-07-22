import argparse
from collections import Counter
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54

from utils import load_trees


def preprocess(counter):
    if isinstance(counter, Counter):
        c = [(k, v) for k, v in counter.items() if len(k.rhs()) == 2]
        c = sorted(c, key=lambda x: x[1], reverse=True)
    elif isinstance(counter, list):
        pass
    return c


def count_productions(trees):
    counter = Counter()
    for t in trees:
        for p in t.productions():
            if not isinstance(p.rhs()[0], str):
                counter[p] += 1
    return counter


def load_count(paths):
    trees = [load_trees(o) for o in paths]
    counts = [
        count_productions([t["tree"] for t in ts]).most_common()
        for ts in trees
    ]
    length = len(counts)

    avgs = []
    for count in zip_longest(*counts):
        avg = sum([c[1] if c is not None else 0 for c in count]) / length
        avgs.append(avg)
    return avgs


def main(
    orgs_path,
    ours_path,
    out_path,
    threshold=200,
    log=False,
    blabel="Original",
    olabel="Ours",
    subgraph=False,
    label_fontsize=17,
    tick_fontsize=13,
    normalize=False,
):
    org_avg_counts = load_count(orgs_path)
    our_avg_counts = load_count(ours_path)

    if threshold > 0:
        org_count = np.array(org_avg_counts[:threshold])
        ours_count = np.array(our_avg_counts[:threshold])
    else:
        org_count = np.array(org_avg_counts)
        ours_count = np.array(our_avg_counts)

    org_x = np.arange(len(org_count)) + 1
    ours_x = np.arange(len(ours_count)) + 1

    # Normalize
    if normalize:
        org_count = org_count / org_count.sum()
        ours_count = ours_count / ours_count.sum()

    diff_count = org_count - np.pad(
        ours_count,
        (0, len(org_count) - len(ours_count)),
        "constant",
        constant_values=0,
    )
    diff_idx = np.arange(len(diff_count)) + 1

    points = np.stack([diff_idx, diff_count], axis=-1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    diff_color = ListedColormap(["tab:orange", "tab:blue"])
    diff_norm = BoundaryNorm([-1, 0, 1], 2)

    fig, ax = plt.subplots(figsize=(6, 3))
    (org,) = ax.plot(org_x, org_count, label=blabel, linewidth=1.3)
    (ours,) = ax.plot(
        ours_x, ours_count, label=olabel, linestyle="--", linewidth=1.3
    )

    # Add difference between two models
    lc = LineCollection(
        segments, cmap=diff_color, norm=diff_norm, linewidth=1, label="diff"
    )
    lc.set_array(diff_count[:-1])

    if log:
        diff_ax = ax.twinx()
        tmp_ax = diff_ax
    else:
        tmp_ax = ax

    tmp_ax.add_collection(lc)
    pos_fill = tmp_ax.fill_between(
        diff_idx,
        diff_count,
        where=diff_count > 0,
        interpolate=True,
        color="tab:blue",
        alpha=0.5,
    )
    neg_fill = tmp_ax.fill_between(
        diff_idx,
        diff_count,
        where=diff_count <= 0,
        interpolate=True,
        color="tab:orange",
        alpha=0.5,
    )

    # Add legend
    ax.legend(
        [org, ours, (pos_fill, neg_fill)],
        [blabel, olabel, "Difference"],
        handler_map={
            (pos_fill, neg_fill): HandlerTuple(ndivide=None, pad=0.0)
        },
        fontsize=tick_fontsize,
        alignment="left",
        handlelength=1.5,
        handletextpad=0.5,
    )

    # Add labels
    ax.set_ylabel("Frequency", fontsize=label_fontsize, labelpad=10)
    ax.set_xlabel("Rank", fontsize=label_fontsize, labelpad=10)
    # # Set yticks
    yticklabels = [l._text for l in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, fontsize=tick_fontsize)
    # # Set xticks
    xticklabels = [l._text for l in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, fontsize=tick_fontsize)
    # # Set additional y-axis
    if log:
        ax.set_yscale("log")
        diff_ax.set_ylabel("Difference", fontsize=label_fontsize, labelpad=10)
        diff_yticklabels = [l._text for l in diff_ax.get_yticklabels()]
        diff_ax.set_yticklabels(diff_yticklabels, fontsize=tick_fontsize)
    ax.grid()

    if subgraph:
        sub = plt.axes([0.3, 0.4, 0.2, 0.4])
        th = 10
        sub_diff_idx = diff_idx[:th]
        sub_diff_count = diff_count[:th]
        sub_lc = LineCollection(
            segments[:th], cmap=diff_color, norm=diff_norm, linewidth=1
        )
        sub_lc.set_array(sub_diff_count[:-1])
        sub.add_collection(sub_lc)
        _ = sub.fill_between(
            sub_diff_idx,
            sub_diff_count,
            where=sub_diff_count > 0,
            interpolate=True,
            color="tab:blue",
            alpha=0.5,
        )
        _ = sub.fill_between(
            sub_diff_idx,
            sub_diff_count,
            where=sub_diff_count <= 0,
            interpolate=True,
            color="tab:orange",
            alpha=0.5,
        )

    # Save figure
    if threshold is not None:
        filename = f"{out_path}_{threshold}"
    else:
        filename = f"{out_path}_total"
    if log:
        filename += "_log"
    plt.savefig(filename + ".png", bbox_inches="tight")
    plt.savefig(filename + ".svg", format="svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orgs", nargs="+", required=True)
    parser.add_argument("--ours", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", default=200, type=int)
    parser.add_argument("--log", default=False, action="store_true")
    parser.add_argument("--blabel", default="Original")
    parser.add_argument("--olabel", default="Ours")
    parser.add_argument("--subgraph", default=False, action="store_true")
    args = parser.parse_args()

    main(
        args.orgs,
        args.ours,
        args.out,
        args.threshold,
        args.log,
        args.blabel,
        args.olabel,
        args.subgraph,
    )

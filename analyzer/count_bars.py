import argparse
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54


def preprocess(counter):
    if isinstance(counter, Counter):
        c = [(k, v) for k, v in counter.items() if len(k.rhs()) == 2]
        c = sorted(c, key=lambda x: x[1], reverse=True)
    elif isinstance(counter, list):
        pass
    return c


def main(org_path, ours_path, out_path, threshold=200, log=False):
    org = pickle.load(open(org_path, "rb"))
    ours = pickle.load(open(ours_path, "rb"))
    # org = preprocess(org)
    # ours = preprocess(ours)

    if threshold is not None:
        org_count = np.array([p[1] for p in org][:threshold])
        ours_count = np.array([p[1] for p in ours][:threshold])
        org_x = np.arange(len(org_count))[:threshold] + 1
        ours_x = np.arange(len(ours_count))[:threshold] + 1
    else:
        org_count = np.array([p[1] for p in org])
        ours_count = np.array([p[1] for p in ours])
        org_x = np.arange(len(org_count)) + 1
        ours_x = np.arange(len(ours_count)) + 1

    # diff_count = org_count - ours_count
    # diff_count = org_count[: len(ours_x)] - ours_count
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
    diff_norm = BoundaryNorm([-1, 1], 2)

    fig, ax = plt.subplots(figsize=(6, 3))
    (org,) = ax.plot(org_x, org_count, label="Neural PCFG")
    (ours,) = ax.plot(ours_x, ours_count, label="Ours")

    if log:
        diff_ax = ax.twinx()

    lc = LineCollection(
        segments, cmap=diff_color, norm=diff_norm, linewidth=1, label="diff"
    )
    lc.set_array(diff_count[:-1])
    if log:
        diff_ax.add_collections(lc)
        pos_fill = diff_ax.fill_between(
            diff_idx,
            diff_count,
            where=diff_count > 0,
            interpolate=True,
            color="tab:blue",
            alpha=0.5,
        )
        neg_fill = diff_ax.fill_between(
            diff_idx,
            diff_count,
            where=diff_count <= 0,
            interpolate=True,
            color="tab:orange",
            alpha=0.5,
        )
    else:
        ax.add_collection(lc)

        pos_fill = ax.fill_between(
            diff_idx,
            diff_count,
            where=diff_count > 0,
            interpolate=True,
            color="tab:blue",
            alpha=0.5,
        )
        neg_fill = ax.fill_between(
            diff_idx,
            diff_count,
            where=diff_count <= 0,
            interpolate=True,
            color="tab:orange",
            alpha=0.5,
        )

    ax.legend(
        [org, ours, (pos_fill, neg_fill)],
        ["Neural PCFG", "Ours", "Difference"],
        handler_map={
            (pos_fill, neg_fill): HandlerTuple(ndivide=None, pad=0.0)
        },
        fontsize=5 / cm,
        alignment="left",
        handlelength=1.5,
        handletextpad=0.5,
    )
    # ax.legend()
    ax.set_ylabel("Frequency", fontsize=7 / cm, labelpad=10)
    ax.set_xlabel("Rank", fontsize=7 / cm, labelpad=10)
    if log:
        ax.set_yscale("log")
        diff_ax.set_ylabel("Difference")
    ax.grid()

    # out_path = Path(out_path).stem
    if threshold is not None:
        filename = f"{out_path}_{threshold}"
    else:
        filename = f"{out_path}_total"
    if log:
        filename += "_log"

    plt.xticks(fontsize=5 / cm)
    plt.yticks(fontsize=5 / cm)
    plt.savefig(filename + ".png", bbox_inches="tight")
    plt.savefig(filename + ".svg", format="svg", bbox_inches="tight")
    plt.close()


# fig, ax = plt.subplots()
# diff_ax = ax.twinx()

# (org,) = ax.plot(org_x, org_count, label="Neural PCFG")
# (ours,) = ax.plot(ours_x, ours_count, label="Ours")

# lc = LineCollection(
#     segments, cmap=diff_color, norm=diff_norm, linewidth=1, label="diff"
# )
# lc.set_array(diff_count[:-1])
# diff_ax.add_collection(lc)

# pos_fill = diff_ax.fill_between(
#     diff_idx,
#     diff_count,
#     where=diff_count > 0,
#     interpolate=True,
#     color="tab:blue",
#     alpha=0.5,
# )
# neg_fill = diff_ax.fill_between(
#     diff_idx,
#     diff_count,
#     where=diff_count <= 0,
#     interpolate=True,
#     color="tab:orange",
#     alpha=0.5,
# )
# ax.legend(
#     [org, ours, (pos_fill, neg_fill)],
#     ["NeuralPCFG", "Ours", "Difference"],
#     handler_map={(pos_fill, neg_fill): HandlerTuple(ndivide=None, pad=0.0)},
# )
# # ax.legend()
# ax.set_ylabel("Frequency")
# ax.set_xlabel("Rank")
# ax.set_yscale("log")
# ax.grid()

# diff_ax.set_ylabel("Difference")

# plt.savefig("count_linear_total_log.png", bbox_inches="tight")
# plt.savefig("count_linear_total_log.svg", format="svg", bbox_inches="tight")
# plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", default=200, type=int)
    parser.add_argument("--log", default=False, action="store_true")
    args = parser.parse_args()

    main(args.org, args.ours, args.out, args.threshold, args.log)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["svg.fonttype"] = "none"

org = pickle.load(open("org_binary_count.pkl", "rb"))
ours = pickle.load(open("ours_binary_count.pkl", "rb"))

threshold = 200
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

fig, ax = plt.subplots()
(org,) = ax.plot(org_x, org_count, label="Neural PCFG")
(ours,) = ax.plot(ours_x, ours_count, label="Ours")

lc = LineCollection(
    segments, cmap=diff_color, norm=diff_norm, linewidth=1, label="diff"
)
lc.set_array(diff_count[:-1])
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
    ["NeuralPCFG", "Ours", "Difference"],
    handler_map={(pos_fill, neg_fill): HandlerTuple(ndivide=None, pad=0.0)},
)
# ax.legend()
ax.set_ylabel("Frequency")
ax.set_xlabel("Rank")
ax.grid()

if threshold is not None:
    filename = f"results/count_linear_{threshold}"
else:
    filename = "results/count_linear_total"
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

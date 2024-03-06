import numpy as np
import torch
from matplotlib import pyplot as plt

from easydict import EasyDict as edict
from parser.model.MFTNPCFG import MFTNPCFG
from parser.model.NeuralPCFG import NeuralPCFG

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54

pt = torch.load(
    "log/n_english_std_baseline/NeuralPCFG2023-12-28-06_03_22/best.pt",
    mmap="cpu",
)["model"]
args = edict({"s_dim": 256})
# model = MFTNPCFG(args)
model = NeuralPCFG(args)
model.load_state_dict(pt)
del pt
model.to("cpu")
rule = model.forward({"word": torch.randint(10000, (1, 20))})
del model

binary = rule["rule"][0].reshape(30, -1).exp()
binary = binary.sort(-1, descending=True)[0]


def threshold_index(thr):
    idx = []
    for b in binary:
        tot = 0
        thr_idx = None
        for i, e in enumerate(b):
            tot += e
            if tot >= thr:
                thr_idx = i + 1
                break
        idx.append(thr_idx)
    return idx


n_rules = threshold_index(0.95)
lim_idx = threshold_index(0.99)

binary = binary.detach().numpy()
NT, r = binary.shape

# n_plot = 4
# # idx_list = np.random.randint(0, NT, n_plot)
# idx_list = [28, 25, 20, 13]

# fig = plt.figure(figsize=(6, 5))
# gs = fig.add_gridspec(
#     n_plot, 1, height_ratios=(1, 1, 1, 1), wspace=0.05, hspace=0.7
# )
# axs = gs.subplots()

# thr = max([lim_idx[i] for i in idx_list])

# for i, ax in enumerate(axs.flatten()):
#     idx = idx_list[i]
#     # thr = lim_idx[idx] if lim_idx[idx] != 1 else 50
#     ax.plot(np.arange(0, thr), binary[idx][:thr])
#     ax.vlines(
#         n_rules[idx],
#         0,
#         1,
#         transform=ax.get_xaxis_transform(),
#         colors="r",
#         linestyles="dashed",
#     )
#     ax.vlines(
#         lim_idx[idx],
#         0,
#         1,
#         transform=ax.get_xaxis_transform(),
#         colors="g",
#         linestyles="dashed",
#     )
#     ax.grid(True, alpha=0.5)
#     ax.set_ylim(0, 1)
#     ax.set_title(f"Nonterminal index: {idx}", fontsize="small", loc="left")
#     xt = ax.get_xticklabels()
#     ax.set_xticklabels(xt, fontsize="small")
#     yt = ax.get_yticklabels()
#     ax.set_yticklabels(yt, fontsize="small")

# fig.supylabel("Probability", fontsize=6 / cm)
# fig.supxlabel("Rank", fontsize=6 / cm)

# filename = f"results/idx{idx}_rule_dist_with_{thr}"
# plt.savefig(f"{filename}.png", bbox_inches="tight")
# plt.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
# plt.close()

# thr = np.max(lim_idx)
# prob_max = binary.flatten().max()

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(5, 6, wspace=0.3, hspace=0.3)
# gs.tight_layout(fig)
axs = gs.subplots()
for i, ax in enumerate(axs.flatten()):
    thr = lim_idx[i] if lim_idx[i] != 1 else 50
    ax.plot(np.arange(0, thr), binary[i][:thr])
    ax.vlines(
        n_rules[i],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        colors="r",
        linestyles="dashed",
    )
    ax.vlines(
        lim_idx[i],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        colors="g",
        linestyles="dashed",
    )
    # lim = ax.get_xlim()
    # ax.set_xticks(list(ax.get_xticks()) + [n_rules[i]])
    ax.set_xlim(0, thr)
    # ax.set_ylim(0, prob_max)
    # ax.set_xlabel("Rank")
    # ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.5)

# fig.tight_layout()
fig.supylabel("Probability", fontsize=6 / cm)
fig.supxlabel("Rank", fontsize=6 / cm)

filename = f"results/tmp_test_rule_dist_with_{thr}"
plt.savefig(f"{filename}.png", bbox_inches="tight")
plt.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
plt.close()

# y_mean = binary.mean(0)
# y_std = binary.std(0)
# rank_mean = np.array(n_rules).mean()
# rank_std = np.array(n_rules).std()

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.plot(np.arange(1, thr + 1), y_mean[:thr])
# ax.fill_between(
#     np.arange(1, thr + 1),
#     y_mean[:thr] - y_std[:thr],
#     y_mean[:thr] + y_std[:thr],
#     alpha=0.3,
# )
# ax.vlines(
#     rank_mean,
#     0,
#     1,
#     transform=ax.get_xaxis_transform(),
#     colors="r",
#     linestyles="dashed",
# )

# ax.axvspan(rank_mean - rank_std, rank_mean + rank_std, alpha=0.3, color="red")

# ax.set_xlabel("rank")
# ax.set_ylabel("probability")
# ax.grid(True, alpha=0.5)

# filename = f"results/rule_dist_with_{thr}"
# plt.savefig(f"{filename}.png", bbox_inches="tight")
# plt.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
# plt.close()

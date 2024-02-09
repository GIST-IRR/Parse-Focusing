from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams["svg.fonttype"] = "none"


with Path("cuc_bins_FTN4500.pkl").open("rb") as f:
    cuc_bins = pickle.load(f)

n_bins = 16
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

fig, ax = plt.subplots()
bottom = np.zeros(n_bins)
for i, cuc in enumerate(weighted_cuc_ox.T):
    label = "False positive" if i == 0 else "True positive"
    p = ax.bar(np.arange(n_bins) + 1, cuc, label=label, bottom=bottom)
    bottom += cuc
    ax.bar_label(p, label_type="center")
ax.legend()
plt.savefig("results/cuc_hist.png", bbox_inches="tight")
plt.savefig("results/cuc_hist.svg", format="svg", bbox_inches="tight")
plt.close()

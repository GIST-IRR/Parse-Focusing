from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54

# Load data
with Path("analyzer/cache/cuc_bins_FTN4500.pkl").open("rb") as f:
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

fig, ax = plt.subplots(figsize=(6, 3))
bottom = np.zeros(n_bins)
for i, cuc in enumerate(weighted_cuc_ox.T):
    label = "Not in gold" if i == 0 else "In gold"
    p = ax.bar(np.arange(n_bins) + 1, cuc, label=label, bottom=bottom)
    bottom += cuc
    # ax.bar_label(p, label_type="center")
ax.legend(fontsize=5 / cm)
ax.set_xlabel("Number of parsers sharing common span", fontsize=6 / cm)
ax.set_ylabel("Count of common spans", fontsize=6 / cm)
plt.xlim(0, n_bins + 1)
xt = list(range(0, n_bins + 2, 2))
xt_label = [str(int(x)) for x in xt]
xt_label[0] = xt_label[0] + "\nSpecific"
xt_label[-1] = xt_label[-1] + "\nCommon"
# xt_label.append("\nCommon")
# xt = np.insert(xt, 1, 1)
# xt = np.append(xt, 17)
# xt_f = str(int(xt[0])) + "\nSpecific"
# xt_l = str(int(xt[-1])) + "\nCommon"
# xt_new = [xt_f] + [str(int(x)) for x in xt[1:-1]] + [xt_l]
# ax.tick_params(axis="both", which="major", labelsize=6 * cm)
plt.yticks(fontsize=5 / cm)
plt.xticks(xt, xt_label, fontsize=5 / cm)
plt.savefig("results/cuc_hist.png", bbox_inches="tight")
plt.savefig("results/cuc_hist.svg", format="svg", bbox_inches="tight")
plt.close()

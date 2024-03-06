import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54
homo_x = ["(SF,SF)", "(NBL,NBL)", "(FGG,FGG)", "(DIORA,DIORA)"]
homo_y = [59.8, 65.0, 68.6, 53.5]
homo_base_y = [52.9, 62.1, 66.1, 44.1]

# for i, v in enumerate(homo_y):
#     homo_y[i] = v - homo_base_y[i]

homo_yerr = [0.6, 0, 0.5, 0]
hetero_x = [
    "(SF,NBL)",
    "(SF,FGG)",
    "(SF,DIORA)",
    "(NBL,FGG)",
    "(NBL,DIORA)",
    "(FGG,DIORA)",
    "(SF,NBL,FGG)",
    "(SF,NBL,DIORA)",
    "(SF,FGG,DIORA)",
    "(NBL,FGG,DIORA)",
    "(SF,NBL,FGG,DIORA)",
]
hetero_y = [65.0, 65.7, 59.1, 69.2, 66.1, 65.1, 64.3, 69.7, 66.3, 68.5, 69.4]
hetero_base_y = [
    59.0,
    60.8,
    50.2,
    64.1,
    53.5,
    55.3,
    61.3,
    54.2,
    55.4,
    57.7,
    57.2,
]

diff_homo = []
diff_hetero = []
for i, v in enumerate(homo_y):
    diff_homo.append(v - homo_base_y[i])
for i, v in enumerate(hetero_y):
    diff_hetero.append(v - hetero_base_y[i])

hetero_yerr = [1.7, 0.3, 0.4, 0.2, 1.5, 0.9, 1.1, 0.9, 0.7, 1.6, 0]

x = homo_x + hetero_x
y = homo_y + hetero_y
yerr = homo_yerr + hetero_yerr

fig, ax = plt.subplots(1, 1, figsize=(6, 2))

# Normal
# fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
# p = axs[0].bar(x, y, yerr=yerr, capsize=4)
# for b in list(p)[len(homo_x) :]:
#     b.set_color("tab:orange")
# axs[0].set_ylim(50, 71)
# axs[0].set_ylabel("F1 score", fontsize=6 / cm)
# axs[0].grid(alpha=0.3, axis="y")
# axs[0].set_title("Total score of multiple parser")
# p = ax.bar(x, y, yerr=yerr, capsize=4)
# for b in list(p)[len(homo_x) :]:
#     b.set_color("tab:orange")
# ax.set_ylim(50, 71)
# ax.set_ylabel("F1 score", fontsize=6 / cm)
# ax.grid(alpha=0.3, axis="y")
# ax.set_title("Total score of multiple parser")

# Diff

diff = diff_homo + diff_hetero
# p = axs[1].bar(x, diff, yerr=yerr, capsize=4)
# for b in list(p)[len(diff_homo) :]:
#     b.set_color("tab:orange")
# axs[1].set_ylabel("Difference of F1", fontsize=6 / cm)
# axs[1].grid(alpha=0.3, axis="y")
# axs[1].set_title(
#     "Difference between multiple parse and mean of pre-trained parser"
# )
p = ax.bar(x, diff, yerr=yerr, capsize=4)
for b in list(p)[len(diff_homo) :]:
    b.set_color("tab:orange")
ax.set_ylabel("Difference in S-F1", fontsize=5 / cm)
ax.grid(alpha=0.3, axis="y")
# ax.set_title(
#     "Difference between multiple parse and mean of pre-trained parser"
# )

plt.xticks(rotation=45, ha="right")
plt.savefig("results/homo_hetero_diff.png", bbox_inches="tight")
plt.savefig("results/homo_hetero_diff.svg", format="svg", bbox_inches="tight")
plt.close()

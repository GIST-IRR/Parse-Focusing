import matplotlib.pyplot as plt
import scipy.stats as stats
from csv import reader

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54

# some random data
f1 = []
ll = []
with open("analyzer/cache/ftn_nt4500_f1_ll.csv", "r") as f:
    rd = reader(f)
    for r in rd:
        f1.append(float(r[0]))
        ll.append(float(r[1]))


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)
    ax.grid(alpha=0.3)
    ax.set_xlabel("F1", fontsize=13, labelpad=10)
    ax.set_ylabel("Negative Log Likelihood", fontsize=13, labelpad=10)

    # now determine nice limits by hand:
    # binwidth = 0.25
    # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # lim = (int(xymax / binwidth) + 1) * binwidth

    # bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=8)
    ax_histy.hist(y, bins=8, orientation="horizontal")


# Start with a square Figure.
fig = plt.figure(figsize=(12 * cm, 12 * cm))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
# gs = fig.add_gridspec(
#     2,
#     2,
#     width_ratios=(4, 1),
#     height_ratios=(1, 4),
#     left=0.15,
#     right=0.9,
#     bottom=0.1,
#     top=0.9,
#     wspace=0.05,
#     hspace=0.05,
# )
gs = fig.add_gridspec(
    2, 2, width_ratios=(6, 1), height_ratios=(1, 6), wspace=0.05, hspace=0.05
)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Correlation Coefficient
corr = stats.pearsonr(f1, ll)
ax.text(
    0.07,
    0.15,
    f"$r={corr[0]:.2f}$\n$P={corr[1]:.2f}$",
    transform=ax.transAxes,
    math_fontfamily="dejavuserif",
    fontsize=11,
    verticalalignment="top",
)
# Draw the scatter plot and marginals.
scatter_hist(f1, ll, ax, ax_histx, ax_histy)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.savefig("results/f1_ll_scatter.png", bbox_inches="tight")
plt.savefig("results/f1_ll_scatter.svg", format="svg", bbox_inches="tight")

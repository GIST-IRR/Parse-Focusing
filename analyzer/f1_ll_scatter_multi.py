import matplotlib.pyplot as plt
import scipy.stats as stats
from csv import reader

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"
cm = 1 / 2.54


# some random data
def f1_ll_open(file):
    f1 = []
    ll = []
    with open(file, "r") as f:
        rd = reader(f)
        for r in rd:
            f1.append(float(r[0]))
            ll.append(float(r[1]))
    return f1, ll


# def scatter(x, y, ax):
#     # the scatter plot:
#     ax.scatter(x, y)
#     ax.grid()
#     ax.set_xlabel("F1", fontsize=9, labelpad=10)
#     ax.set_ylabel("Negative Log Likelihood", fontsize=9, labelpad=10)


f1_4500, ll_4500 = f1_ll_open("analyzer/cache/ftn_nt4500_f1_ll.csv")
f1_250, ll_250 = f1_ll_open("analyzer/cache/ftn_nt250_f1_ll.csv")
f1_30, ll_30 = f1_ll_open("analyzer/cache/ftn_nt30_f1_ll.csv")

# Start with a square Figure.
fig = plt.figure(figsize=(12 * cm, 12 * cm))
# Create the Axes.
ax = fig.add_subplot()
# Correlation Coefficient
corr_4500 = stats.pearsonr(f1_4500, ll_4500)
corr_250 = stats.pearsonr(f1_250, ll_250)
corr_30 = stats.pearsonr(f1_30, ll_30)

# ax.text(
#     0.07,
#     0.15,
#     f"$r={corr[0]:.2f}$\n$P={corr[1]:.2f}$",
#     transform=ax.transAxes,
#     fontsize=10,
#     verticalalignment="top",
# )
# Draw the scatter plot and marginals.
# the scatter plot:
ax.scatter(f1_4500, ll_4500, label="NT=4500", s=10)
ax.scatter(f1_250, ll_250, label="NT=250", s=10)
ax.scatter(f1_30, ll_30, label="NT=30", s=10)

ax.grid(alpha=0.5)
ax.set_xlabel("F1", fontsize=13, labelpad=10)
ax.set_ylabel("Negative Log Likelihood (NLL)", fontsize=13, labelpad=10)
ax.legend(fontsize=11, handletextpad=0.0, borderpad=0.6)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.savefig("results/f1_ll_scatter_multi.png", bbox_inches="tight")
plt.savefig(
    "results/f1_ll_scatter_multi.svg", format="svg", bbox_inches="tight"
)

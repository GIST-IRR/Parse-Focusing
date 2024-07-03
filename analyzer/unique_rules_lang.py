import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"


def draw_figure(file_path, name, ax):
    data = pd.read_csv(file_path)
    data = data.apply(pd.to_numeric, errors="ignore")

    for index, row in data.iterrows():
        model_name = row["model name"]
        x = data.columns[1:].astype(
            int
        )  # 첫 번째 열을 제외한 나머지 열이 sentence length
        y = row[1:].values  # 첫 번째 열을 제외한 나머지 열이 rule type의 수
        ax.plot(
            x,
            y,
            label=model_name,
            # linestyle=linestyles[index],
            # color=linecolors[index],
        )

    # 차트 제목과 축 레이블을 설정합니다.
    # plt.title("Sentence Length vs Rule Types")
    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=17)
    ax.legend(fontsize=17)
    ax.grid(True)
    ax.set_title(name, fontsize=25)


file_pathes = [
    "fig_rule_util/bas_rule_util.csv",
    "fig_rule_util/chi_rule_util.csv",
    "fig_rule_util/eng_rule_util.csv",
    "fig_rule_util/fre_rule_util.csv",
    "fig_rule_util/ger_rule_util.csv",
    "fig_rule_util/heb_rule_util.csv",
    "fig_rule_util/hun_rule_util.csv",
    "fig_rule_util/kor_rule_util.csv",
    "fig_rule_util/pol_rule_util.csv",
    "fig_rule_util/swe_rule_util.csv",
]
titles = [
    "Basque",
    "Chinese",
    "English",
    "French",
    "German",
    "Hebrew",
    "Hungarian",
    "Korean",
    "Polish",
    "Swedish",
]

fig, axes = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True)
for path, title, ax in zip(file_pathes, titles, axes.flatten()):
    draw_figure(path, title, ax)

fontsize = 30
fig.supxlabel("Sentence Length", fontsize=fontsize)
fig.supylabel("Number of Rule Types", fontsize=fontsize)
plt.savefig("rule_util.png", bbox_inches="tight")
plt.savefig("rule_util.svg", format="svg", bbox_inches="tight")

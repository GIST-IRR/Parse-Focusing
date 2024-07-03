import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"


# Read the CSV file.
file_path = "fig_rule_util/eng_rule_util.csv"
data = pd.read_csv(file_path)

# Convert the remaining data to numeric, excluding the model name
data = data.apply(pd.to_numeric, errors="ignore")

# Draw line charts for each model.
plt.figure(figsize=(6, 6))

linecolors = ["#ff7f0e", "#2ca02c", "#d62728", "#ff7f0e", "#2ca02c", "#d62728"]
linestyles = ["solid", "solid", "solid", "dashed", "dashed", "dashed"]

for index, row in data.iterrows():
    model_name = row["model name"]
    x = data.columns[1:].astype(
        int
    )  # 첫 번째 열을 제외한 나머지 열이 sentence length
    y = row[1:].values  # 첫 번째 열을 제외한 나머지 열이 rule type의 수
    plt.plot(
        x,
        y,
        label=model_name,
        # linestyle=linestyles[index],
        # color=linecolors[index],
    )

# 차트 제목과 축 레이블을 설정합니다.
fontsize = 19
plt.legend(fontsize=15)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Sentence Length", fontsize=fontsize)
plt.ylabel("Number of Rule Types", fontsize=fontsize)
plt.savefig("eng_rule_util.png", bbox_inches="tight")
plt.savefig("eng_rule_util.svg", format="svg", bbox_inches="tight")

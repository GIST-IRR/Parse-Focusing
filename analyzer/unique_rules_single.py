import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Times New Roman"


def main(file_path, output_path):
    file_path = (
        "fig_rule_util/eng_rule_util.csv"  # 여기에 CSV 파일 경로를 입력하세요.
    )
    data = pd.read_csv(file_path)

    # 모델 이름을 제외한 나머지 데이터를 숫자로 변환
    data = data.apply(pd.to_numeric, errors="ignore")

    # 각 모델에 대해 라인 차트를 그립니다.
    plt.figure(figsize=(6, 6))

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


if __name__ == "__main__":
    main("fig_rule_util/eng_rule_util.csv", "eng_rule_util.png")

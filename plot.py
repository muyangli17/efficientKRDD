import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 注册本地字体文件（将 .ttf 文件放入 fonts/ 目录即可）
font_dir = os.path.join(os.path.dirname(__file__), "fonts")
if os.path.isdir(font_dir):
    for font_file in os.listdir(font_dir):
        if font_file.lower().endswith((".ttf", ".otf")):
            fm.fontManager.addfont(os.path.join(font_dir, font_file))

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # 数学字体与 Times New Roman 风格一致

ipc = [1, 10, 50]
# 实验结果：
#   old = 旧模型（baseline），new = 新模型（提出的方法）
#   acc = 准确率 (%)，time = 单次运行时间 (s)
results = {
    "MNIST": {
        "old_acc": [88.1, 96.8, 98.4],
        "old_time": [22055, 19984, 19647],
        "new_acc": [77.66, 96.59, 98.10],
        "new_time": [7884, 8773, 9370],
    },
    "FashionMNIST": {
        "old_acc": [72.2, 83.3, 88.6],
        "old_time": [19731, 19309, 19901],
        "new_acc": [67.80, 83.28, 87.77],
        "new_time": [8697, 9162, 8932],
    },
    "SVHN": {
        "old_acc": [21.6, 72.7, 80.4],
        "old_time": [19965, 19786, 19972],
        "new_acc": [22.01, 72.45, 82.22],
        "new_time": [8625, 8834, 9272],
    },
    "CIFAR10": {
        "old_acc": [26.2, 48.5, 60.9],
        "old_time": [19381, 17282, 17116],
        "new_acc": [27.08, 45.72, 58.16],
        "new_time": [8718, 8586, 8447],
    },
    # "CIFAR100": {
    #     "old_acc": [11.5, 29.5, 40.2],
    #     "old_time": [placeholder_time, placeholder_time, placeholder_time],
    #     "new_acc": [placeholder_acc, placeholder_acc, placeholder_acc],
    #     "new_time": [placeholder_time, placeholder_time, placeholder_time],
    # },
}

# ---------------------------------------------------------------------------
# 可视化设计说明：
#   - X 轴：IPC (Images Per Class)，连续自变量
#   - Y 轴左（折线）：准确率 (Accuracy %)
#     准确率随 IPC 增加呈单调上升趋势，折线图能清晰展示这一规律，
#     便于直接比较新旧模型在不同 IPC 下的精度差异。
#   - Y 轴右（柱状图）：运行时间 (Runtime s)
#     在相同 IPC 位置绘制新旧模型的时间柱状图，右侧轴标注时间刻度。
# ---------------------------------------------------------------------------

if not os.path.exists("./plot"):
    os.makedirs("./plot")

for dataset in results.keys():
    # ======================== 绘图 ========================
    fig, ax1 = plt.subplots(figsize=(6, 6))

    old_time = results[dataset]["old_time"]
    new_time = results[dataset]["new_time"]
    old_acc = results[dataset]["old_acc"]
    new_acc = results[dataset]["new_acc"]

    # ---- X 轴位置：等间隔排列，刻标签示 IPC 值 ----
    x_pos = np.arange(len(ipc))  # [0, 1, 2] 等间距

    # ---- 折线图（左轴）：展示准确率随 IPC 的变化趋势 ----
    ax1.plot(
        x_pos,
        old_acc,
        color="blue",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        marker="o",
        markersize=6,
        label="Vanila KR (Acc)",
    )
    ax1.plot(
        x_pos,
        new_acc,
        color="red",
        linestyle="-",
        alpha=0.8,
        linewidth=1.5,
        marker="s",
        markersize=6,
        label="Ours (Acc)",
    )

    ax1.set_xlabel("IPC (Images Per Class)", fontsize=16)
    ax1.set_ylabel("Accuracy (%)", fontsize=16, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ipc)
    ax1.set_ylim(bottom=0)

    # ---- 柱状图（右轴）：在相同 xtick 处绘制时间柱状图 ----
    ax2 = ax1.twinx()

    bar_width = 0.2  # 每组两个柱子的总宽度（等间距下适当减小）

    bars_old = ax2.bar(
        x_pos - bar_width / 2,
        old_time,
        width=bar_width,
        color="blue",
        alpha=0.35,
        edgecolor="blue",
        linewidth=0.8,
        label="Vanila KR (Time)",
    )
    bars_new = ax2.bar(
        x_pos + bar_width / 2,
        new_time,
        width=bar_width,
        color="red",
        alpha=0.35,
        edgecolor="red",
        linewidth=0.8,
        label="Ours (Time)",
    )

    ax2.set_ylabel("Runtime (s)", fontsize=16, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # ---- 在柱状图上方标注时间数值 ----
    for bar, t in zip(bars_old, old_time):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t}s",
            ha="center",
            va="bottom",
            fontsize=10,
            color="blue",
            alpha=0.85,
        )
    for bar, t in zip(bars_new, new_time):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t}s",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            alpha=0.85,
        )

    # ======================== 图形美化 ========================
    plt.title(
        f"{dataset} — Accuracy (line) & Runtime (bar)",
        fontsize=18,
    )

    # 图例：合并左右两轴的图例句柄，共 4 个词条
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    ax1.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()

    # 保存图像（300 DPI 适合论文排版）
    plt.savefig(f"./plot/{dataset}.png", dpi=300)
    plt.close()  # 显式关闭图形，释放内存

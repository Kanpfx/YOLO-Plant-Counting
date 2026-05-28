"""绘制部署结果混淆矩阵热力图。"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体设置
for _font in ("SimSun", "SimHei", "Microsoft YaHei", "Source Han Sans CN", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue

# ── 参数 ───────────────────────────────────────────────────────────────
CLASS_NAMES = ("玉米", "甜菜", "水稻", "背景")

# ── 数据读取 ──────────────────────────────────────────────────────────
def load_confusion_matrix(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split(",")[1:]  # skip "gt\pred"
    n = len(header)
    mat = np.zeros((n, n), dtype=np.int32)
    for i, line in enumerate(lines[1:]):
        parts = line.split(",")
        mat[i] = [int(x) for x in parts[1:]]
    return mat, header

# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "deploy", "jetson_confusion_matrix.csv"
    )
    mat, _ = load_confusion_matrix(csv_path)

    # 归一化为行百分比（每行除该行总和）
    row_sums = mat.sum(axis=1, keepdims=True, dtype=np.float32)
    row_sums[row_sums == 0] = 1
    mat_norm = mat.astype(np.float32) / row_sums

    n = len(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # 绘制热力图
    im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)

    # 在每个格子写入数值和百分比
    for i in range(n):
        for j in range(n):
            count = mat[i, j]
            pct = mat_norm[i, j]
            if count > 0:
                text = f"{count}\n({pct:.1%})"
            else:
                text = "0"
            color = "white" if pct > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=15, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, fontsize=16, fontweight="bold")
    ax.set_yticklabels(CLASS_NAMES, fontsize=16, fontweight="bold")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    ax.set_xlabel("预测类别", fontsize=16, fontweight="bold")
    ax.set_ylabel("真实类别", fontsize=16, fontweight="bold")

    # 细边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.35")

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("归一化比例", fontsize=15, fontweight="bold")

    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "images", "部署结果混淆矩阵.png"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存至：{out_path}")

if __name__ == "__main__":
    main()

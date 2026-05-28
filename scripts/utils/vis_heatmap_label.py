"""临时脚本：展示高斯热力图标签的生成过程。"""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 中文字体设置
for _font in ("SimSun", "SimHei", "Microsoft YaHei", "Source Han Sans CN", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue

# ── 参数 ───────────────────────────────────────────────────────────────
PATCH_SIZE = 1024
STRIDES = (4, 8, 16)
NUM_CLASSES = 3
CLASS_NAMES = ("玉米", "甜菜", "水稻")
CLASS_COLORS = [(1.0, 0.2, 0.2), (0.2, 0.7, 0.2), (0.2, 0.2, 1.0)]

# ── 高斯核 ────────────────────────────────────────────────────────────
GAUSSIAN_CACHE = {}

def get_gaussian(radius):
    if radius not in GAUSSIAN_CACHE:
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        sigma = (2 * radius + 1) / 5
        GAUSSIAN_CACHE[radius] = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
    return GAUSSIAN_CACHE[radius]

def draw_gaussian(heatmap, x, y, radius):
    x, y = int(x), int(y)
    h, w = heatmap.shape
    gaussian = get_gaussian(radius)
    left, right = max(0, x - radius), min(w, x + radius + 1)
    top, bottom = max(0, y - radius), min(h, y + radius + 1)
    g_left = left - (x - radius)
    g_right = g_left + (right - left)
    g_top = top - (y - radius)
    g_bottom = g_top + (bottom - top)
    if left < right and top < bottom:
        heatmap[top:bottom, left:right] = np.maximum(
            heatmap[top:bottom, left:right],
            gaussian[g_top:g_bottom, g_left:g_right],
        )

# ── 标签读取 ──────────────────────────────────────────────────────────
def load_boxes(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 5), np.float32)
    with open(label_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if not rows:
        return np.zeros((0, 5), np.float32)
    return np.array([list(map(float, row.split())) for row in rows], np.float32)

# ── 构建所有尺度的 heatmap ────────────────────────────────────────────
def build_all_heatmaps(boxes, strides=(4, 8, 16), num_classes=3, patch_size=1024):
    result = {}
    for stride in strides:
        size = patch_size // stride
        scale = patch_size / stride
        radius_cap = int(64 / stride)
        hm = np.zeros((num_classes, size, size), np.float32)
        for cls, cx, cy, w, h in boxes:
            cls = int(cls)
            if not (0 <= cls < num_classes):
                continue
            x, y = cx * scale, cy * scale
            ix, iy = int(x), int(y)
            if not (0 <= ix < size and 0 <= iy < size):
                continue
            bw_img, bh_img = w * patch_size, h * patch_size
            radius = min(int(max(1, min(bw_img, bh_img) / (2 * stride))), radius_cap)
            draw_gaussian(hm[cls], x, y, radius)
        result[stride] = hm
    return result

# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset")
    train_img = os.path.join(data_root, "train", "images")
    train_lbl = os.path.join(data_root, "train", "labels")

    picks = {
        "水稻": "Rice_001_010.jpg",
        "甜菜": "Sugarbeet_001_010.jpg",
        "玉米": "Maize_002_010.jpg",
    }

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    col_titles = ["原图 + 标注框", "P2 热力图 (s=4)", "P3 热力图 (s=8)", "P4 热力图 (s=16)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=24, fontweight="bold")

    for row_idx, (crop_name, fname) in enumerate(picks.items()):
        img_path = os.path.join(train_img, fname)
        lbl_path = os.path.join(train_lbl, fname.replace(".jpg", ".txt"))

        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img)
        boxes = load_boxes(lbl_path)
        all_hm = build_all_heatmaps(boxes, STRIDES, NUM_CLASSES, PATCH_SIZE)

        # ── 第 1 列：原图 + bbox ──
        ax = axes[row_idx][0]
        ax.imshow(img_arr)
        for cls, cx, cy, w, h in boxes:
            cls = int(cls)
            if not (0 <= cls < NUM_CLASSES):
                continue
            x1 = (cx - w / 2) * PATCH_SIZE
            y1 = (cy - h / 2) * PATCH_SIZE
            rect = mpatches.Rectangle(
                (x1, y1), w * PATCH_SIZE, h * PATCH_SIZE,
                linewidth=2.0, edgecolor=CLASS_COLORS[cls], facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_ylabel(crop_name, fontsize=24, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # ── 第 2-4 列：heatmap RGB 叠加 ──
        for col_idx, stride in enumerate(STRIDES):
            ax = axes[row_idx][col_idx + 1]
            hm = all_hm[stride]
            rgb = np.stack([hm[0], hm[1], hm[2]], axis=-1)
            ax.imshow(rgb)
            ax.set_xticks([]); ax.set_yticks([])

    # 图例
    legend_patches = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=24, frameon=False, bbox_to_anchor=(0.5, 0.02))

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "高斯热力图标签示例.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存至：{out_path}")

if __name__ == "__main__":
    main()

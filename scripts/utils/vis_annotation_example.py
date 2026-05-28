"""绘制数据集标注示例图：每类各选一张原始大图，截取竖条展示标注框。"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

for _font in ("SimSun", "SimHei", "Microsoft YaHei", "Source Han Sans CN", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue

CLASS_NAMES = ("玉米", "甜菜", "水稻")
CLASS_COLORS = [(1.0, 0.2, 0.2), (0.2, 0.7, 0.2), (0.2, 0.2, 1.0)]
STRIP_RATIO = (1, 2)

PICKS = [
    ("玉米", "Maize (2).JPG"),
    ("甜菜", "Sugarbeet (1).JPG"),
    ("水稻", "Rice (1).JPG"),
]

def load_boxes(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 5), np.float32)
    with open(label_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if not rows:
        return np.zeros((0, 5), np.float32)
    return np.array([list(map(float, row.split())) for row in rows], np.float32)

def pick_strip_region(boxes, img_w, img_h, rw=1, rh=2):
    sw = int(img_w * 0.25)
    sh = int(sw * rh / rw)

    if sh > img_h:
        sh = img_h
        sw = int(sh * rw / rh)

    if len(boxes) > 0:
        cx_arr = boxes[:, 1] * img_w
        center_x = int(np.median(cx_arr))
    else:
        center_x = img_w // 2

    x1 = max(0, center_x - sw // 2)
    x2 = min(img_w, x1 + sw)
    x1 = x2 - sw

    y1 = 0
    y2 = min(img_h, sh)
    if y2 - y1 < sh:
        y2 = img_h
        y1 = max(0, img_h - sh)

    return x1, y1, x2, y2

def main():
    base_img = os.path.join(os.path.dirname(__file__), "..", "..", "data", "original", "train", "images")
    base_lbl = os.path.join(os.path.dirname(__file__), "..", "..", "data", "original", "train", "labels")

    fig, axes = plt.subplots(1, 3, figsize=(10, 7.2))

    for idx, (crop_name, fname) in enumerate(PICKS):
        img_path = os.path.join(base_img, fname)
        lbl_path = os.path.join(base_lbl, fname.replace(".JPG", ".txt").replace(".jpg", ".txt"))

        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img)
        img_h, img_w = img_arr.shape[:2]
        boxes = load_boxes(lbl_path)

        x1, y1, x2, y2 = pick_strip_region(boxes, img_w, img_h, *STRIP_RATIO)
        strip = img_arr[y1:y2, x1:x2]

        ax = axes[idx]
        ax.imshow(strip)

        for cls, cx, cy, w, h in boxes:
            cls = int(cls)
            if not (0 <= cls < len(CLASS_NAMES)):
                continue

            px = cx * img_w
            py = cy * img_h
            bw = w * img_w
            bh = h * img_h

            bx1 = px - bw / 2 - x1
            by1 = py - bh / 2 - y1

            if bx1 < -bw or by1 < -bh or bx1 > (x2 - x1) or by1 > (y2 - y1):
                continue

            rect = mpatches.Rectangle(
                (bx1, by1),
                bw,
                bh,
                linewidth=1.8,
                edgecolor=CLASS_COLORS[cls],
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_anchor("S")

    legend_patches = [
        mpatches.Patch(color=c, label=n)
        for c, n in zip(CLASS_COLORS, CLASS_NAMES)
    ]

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.98,
        bottom=0.12,
        wspace=0.04,
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "数据集标注示例图.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存至：{out_path}")

if __name__ == "__main__":
    main()
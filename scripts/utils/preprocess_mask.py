"""颜色阈值与 ExG 阈值预处理脚本。"""

import os

import cv2
import numpy as np
from PIL import Image


MODE = "maize"  # "maize" 或 "exg"
IMG_DIR = "zheng"
LABEL_DIR = "labels"
EXG_IMG_DIR = "Ses1"
EXG_OUT_DIR = "result"

B_LOW, B_HIGH = 140, 200
CR_LOW, CR_HIGH = 130, 185
CB_LOW, CB_HIGH = 75, 140
SMALL_OPEN_KERNEL = 3
BRIDGE_KERNEL = 25
MIN_CLUSTER_AREA = 3000
BBOX_PAD_RATIO = 0.05
CLS_ID = 0


def build_exg_mask(img_path):
    """用 ExG 和 Otsu 阈值提取绿色目标中心点。"""
    img = np.array(Image.open(img_path).convert("RGB"), np.float32)
    height, width = img.shape[:2]
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    exg = 2 * green - red - blue
    p1, p99 = np.percentile(exg, (1, 99))
    exg_u8 = ((np.clip(exg, p1, p99) - p1) / (p99 - p1) * 255).astype(np.uint8)

    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
    _, _, stat, cen = cv2.connectedComponentsWithStats(mask)
    points = [(x, y) for i, (x, y) in enumerate(cen[1:], 1) if stat[i, cv2.CC_STAT_AREA] >= 100]
    return points, height, width


def build_maize_mask(img):
    """用 LAB/YCrCb 颜色阈值生成玉米簇目标框。"""
    height, width = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    _, _, b_channel = cv2.split(lab)
    _, cr_channel, cb_channel = cv2.split(ycrcb)

    mask_b = cv2.inRange(b_channel, B_LOW, B_HIGH)
    mask_cr = cv2.inRange(cr_channel, CR_LOW, CR_HIGH)
    mask_cb = cv2.inRange(cb_channel, CB_LOW, CB_HIGH)
    mask = cv2.bitwise_and(mask_b, mask_cr)
    mask = cv2.bitwise_and(mask, mask_cb)

    # 去噪后膨胀，使同一簇目标合并成一个框。
    if SMALL_OPEN_KERNEL > 0:
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMALL_OPEN_KERNEL, SMALL_OPEN_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BRIDGE_KERNEL, BRIDGE_KERNEL))
    mask_cluster = cv2.dilate(mask, kernel_bridge, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_cluster, connectivity=8)

    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < MIN_CLUSTER_AREA:
            continue

        pad_x = int(BBOX_PAD_RATIO * w)
        pad_y = int(BBOX_PAD_RATIO * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        boxes.append((CLS_ID, cx / width, cy / height, bw / width, bh / height))
    return boxes


def run_exg():
    os.makedirs(EXG_OUT_DIR, exist_ok=True)
    for name in os.listdir(EXG_IMG_DIR):
        if not name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(EXG_IMG_DIR, name)
        points, height, width = build_exg_mask(img_path)
        txt_path = os.path.join(EXG_OUT_DIR, os.path.splitext(name)[0] + ".txt")
        with open(txt_path, "w", newline="\n", encoding="utf-8") as f:
            for x, y in points:
                f.write(f"0 {x / width:.6f} {y / height:.6f} 0.005 {0.005 / height * width}\n")
        print(name, "labels:", len(points))


def run_maize():
    os.makedirs(LABEL_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
    print(f"Found {len(img_files)} images")

    for fname in img_files:
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = build_maize_mask(img)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(fname)[0] + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes))
        print(f"{fname}: {len(boxes)} boxes")

    print("Done.")


def main():
    if MODE == "exg":
        run_exg()
    elif MODE == "maize":
        run_maize()
    else:
        raise ValueError(f"未知预处理模式: {MODE}")


if __name__ == "__main__":
    main()

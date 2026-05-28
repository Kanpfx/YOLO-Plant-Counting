from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_config import CLASS_NAMES, IMG_SIZE, NUM_CLASSES, SCRIPT_DIR


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_code_path():
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def list_images(image_dir: Path):
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def resolve_dataset_dirs(root: Path):
    root = Path(root)
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return root / "images", root / "labels"
    if root.name == "images" and (root.parent / "labels").is_dir():
        return root, root.parent / "labels"
    raise FileNotFoundError(f"数据集目录应包含 images/labels: {root}")


def load_yolo_boxes(label_path: Path, width: int, height: int, num_classes=NUM_CLASSES):
    by_class = defaultdict(list)
    if not label_path.exists():
        return by_class
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split()[:5])
        cls = int(cls)
        if not (0 <= cls < num_classes):
            continue
        x1 = (cx - bw * 0.5) * width
        y1 = (cy - bh * 0.5) * height
        x2 = (cx + bw * 0.5) * width
        y2 = (cy + bh * 0.5) * height
        by_class[cls].append((x1, y1, x2, y2))
    return by_class


def read_image(path: Path, downsample=1, restore_size=False, resize_scale=None):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if resize_scale is not None and float(resize_scale) != 1.0:
        h, w = img.shape[:2]
        scale = float(resize_scale)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=interpolation,
        )
    if downsample and downsample > 1:
        h, w = img.shape[:2]
        scale = float(downsample)
        img = cv2.resize(img, (max(1, int(round(w / scale))), max(1, int(round(h / scale)))), interpolation=cv2.INTER_AREA)
        if restore_size:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def box_iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    pa = np.asarray(pred_boxes, dtype=np.float32)
    gb = np.asarray(gt_boxes, dtype=np.float32)
    x1 = np.maximum(pa[:, None, 0], gb[None, :, 0])
    y1 = np.maximum(pa[:, None, 1], gb[None, :, 1])
    x2 = np.minimum(pa[:, None, 2], gb[None, :, 2])
    y2 = np.minimum(pa[:, None, 3], gb[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(pa[:, 2] - pa[:, 0], 0.0, None) * np.clip(pa[:, 3] - pa[:, 1], 0.0, None)
    area_b = np.clip(gb[:, 2] - gb[:, 0], 0.0, None) * np.clip(gb[:, 3] - gb[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def match_class(pred, gt, iou_thresh=0.5):
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    pred = np.asarray(pred, dtype=np.float32)
    order = np.argsort(-pred[:, 4])
    pred = pred[order]
    gt_boxes = np.asarray(gt, dtype=np.float32)
    iou = box_iou_matrix(pred[:, :4], gt_boxes)
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = fp = 0
    for i in range(len(pred)):
        row = iou[i].copy()
        row[matched] = -1.0
        best = int(row.argmax())
        if float(row[best]) >= iou_thresh and not matched[best]:
            matched[best] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~matched).sum())


def empty_stats():
    return {
        "tp": np.zeros(NUM_CLASSES, dtype=np.int64),
        "fp": np.zeros(NUM_CLASSES, dtype=np.int64),
        "fn": np.zeros(NUM_CLASSES, dtype=np.int64),
        "count_abs": np.zeros(NUM_CLASSES + 1, dtype=np.float64),
        "count_rel": np.zeros(NUM_CLASSES + 1, dtype=np.float64),
        "count_acc": np.zeros(NUM_CLASSES + 1, dtype=np.float64),
        "images": 0,
    }


def update_stats(stats, pred_by_class, gt_by_class, iou_thresh=0.5):
    pred_total = gt_total = 0
    for cls in range(NUM_CLASSES):
        pred = pred_by_class.get(cls, [])
        gt = gt_by_class.get(cls, [])
        tp, fp, fn = match_class(pred, gt, iou_thresh=iou_thresh)
        stats["tp"][cls] += tp
        stats["fp"][cls] += fp
        stats["fn"][cls] += fn

        pc, gc = len(pred), len(gt)
        pred_total += pc
        gt_total += gc
        err = abs(pc - gc)
        rel = err / max(gc, 1)
        stats["count_abs"][cls] += err
        stats["count_rel"][cls] += rel
        stats["count_acc"][cls] += max(0.0, 1.0 - rel)

    err = abs(pred_total - gt_total)
    rel = err / max(gt_total, 1)
    stats["count_abs"][NUM_CLASSES] += err
    stats["count_rel"][NUM_CLASSES] += rel
    stats["count_acc"][NUM_CLASSES] += max(0.0, 1.0 - rel)
    stats["images"] += 1


def summarize_stats(stats, model, display_name, extra=None):
    rows = []
    n = max(int(stats["images"]), 1)
    for cls in list(range(NUM_CLASSES)) + ["all"]:
        idx = NUM_CLASSES if cls == "all" else cls
        if cls == "all":
            tp = int(stats["tp"].sum())
            fp = int(stats["fp"].sum())
            fn = int(stats["fn"].sum())
            class_name = "all"
        else:
            tp = int(stats["tp"][cls])
            fp = int(stats["fp"][cls])
            fn = int(stats["fn"][cls])
            class_name = CLASS_NAMES[cls]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        row = {
            "model": model,
            "name": display_name,
            "class": class_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mae": stats["count_abs"][idx] / n,
            "mre": stats["count_rel"][idx] / n,
            "count_acc": stats["count_acc"][idx] / n,
            "num_images": n,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

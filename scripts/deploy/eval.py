import csv
from pathlib import Path

import numpy as np


PRED_DIR = "result"
GT_DIR = "labels"
OUT_CSV = "jetson_result.csv"
CONFUSION_CSV = "jetson_confusion_matrix.csv"
NUM_CLASSES = 3
IOU_THRESH = 0.5
CLASS_NAMES = ("Maize", "Sugarbeet", "Rice")
BACKGROUND = "background"


def load_yolo_xywh(path):
    boxes = []
    path = Path(path)
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        if 0 <= cls < NUM_CLASSES:
            boxes.append(
                {
                    "cls": cls,
                    "xyxy": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    "score": float(parts[5]) if len(parts) > 5 else 1.0,
                }
            )
    return boxes


def iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    pred = np.asarray(pred_boxes, dtype=np.float32)
    gt = np.asarray(gt_boxes, dtype=np.float32)
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = np.clip(pred[:, 2] - pred[:, 0], 0, None) * np.clip(pred[:, 3] - pred[:, 1], 0, None)
    area_g = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / union, 0)


def match_image(pred, gt):
    """Greedy one-to-one matching across classes for metrics and confusion."""
    pred = sorted(pred, key=lambda item: item["score"], reverse=True)
    pred_boxes = [item["xyxy"] for item in pred]
    gt_boxes = [item["xyxy"] for item in gt]
    ious = iou_matrix(pred_boxes, gt_boxes)
    matched_gt = np.zeros(len(gt), dtype=bool)
    matches = []
    unmatched_pred = []

    for pred_idx, pred_item in enumerate(pred):
        if len(gt) == 0:
            unmatched_pred.append(pred_idx)
            continue
        row = ious[pred_idx].copy()
        row[matched_gt] = -1
        gt_idx = int(row.argmax())
        if row[gt_idx] >= IOU_THRESH:
            matched_gt[gt_idx] = True
            matches.append((pred_idx, gt_idx))
        else:
            unmatched_pred.append(pred_idx)

    unmatched_gt = [idx for idx, matched in enumerate(matched_gt) if not matched]
    return pred, gt, matches, unmatched_pred, unmatched_gt


def empty_stats():
    return {
        "tp": np.zeros(NUM_CLASSES, dtype=np.int64),
        "fp": np.zeros(NUM_CLASSES, dtype=np.int64),
        "fn": np.zeros(NUM_CLASSES, dtype=np.int64),
        "pred": np.zeros(NUM_CLASSES, dtype=np.int64),
        "gt": np.zeros(NUM_CLASSES, dtype=np.int64),
        "count_abs": np.zeros(NUM_CLASSES + 1, dtype=np.float64),
        "count_rel": np.zeros(NUM_CLASSES + 1, dtype=np.float64),
        "images": 0,
        "confusion": np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64),
    }


def update_stats(stats, pred, gt):
    pred, gt, matches, unmatched_pred, unmatched_gt = match_image(pred, gt)
    pred_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    gt_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for item in pred:
        pred_counts[item["cls"]] += 1
    for item in gt:
        gt_counts[item["cls"]] += 1

    stats["pred"] += pred_counts
    stats["gt"] += gt_counts

    for pred_idx, gt_idx in matches:
        pred_cls = pred[pred_idx]["cls"]
        gt_cls = gt[gt_idx]["cls"]
        stats["confusion"][gt_cls, pred_cls] += 1
        if pred_cls == gt_cls:
            stats["tp"][gt_cls] += 1
        else:
            stats["fp"][pred_cls] += 1
            stats["fn"][gt_cls] += 1

    for pred_idx in unmatched_pred:
        pred_cls = pred[pred_idx]["cls"]
        stats["fp"][pred_cls] += 1
        stats["confusion"][NUM_CLASSES, pred_cls] += 1

    for gt_idx in unmatched_gt:
        gt_cls = gt[gt_idx]["cls"]
        stats["fn"][gt_cls] += 1
        stats["confusion"][gt_cls, NUM_CLASSES] += 1

    pred_total = int(pred_counts.sum())
    gt_total = int(gt_counts.sum())
    for cls in range(NUM_CLASSES):
        err = abs(int(pred_counts[cls]) - int(gt_counts[cls]))
        rel = err / max(int(gt_counts[cls]), 1)
        stats["count_abs"][cls] += err
        stats["count_rel"][cls] += rel

    err = abs(pred_total - gt_total)
    rel = err / max(gt_total, 1)
    stats["count_abs"][NUM_CLASSES] += err
    stats["count_rel"][NUM_CLASSES] += rel
    stats["images"] += 1


def metric_row(stats, cls):
    n = max(stats["images"], 1)
    if cls == "all":
        idx = NUM_CLASSES
        tp = int(stats["tp"].sum())
        fp = int(stats["fp"].sum())
        fn = int(stats["fn"].sum())
        pred = int(stats["pred"].sum())
        gt = int(stats["gt"].sum())
    else:
        idx = cls
        tp = int(stats["tp"][cls])
        fp = int(stats["fp"][cls])
        fn = int(stats["fn"][cls])
        pred = int(stats["pred"][cls])
        gt = int(stats["gt"][cls])

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "class": class_name(cls),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": pred,
        "gt_count": gt,
        "mae": stats["count_abs"][idx] / n,
        "mre": stats["count_rel"][idx] / n,
        "num_images": n,
    }


def write_csv(path, rows):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path, matrix):
    labels = list(CLASS_NAMES) + [BACKGROUND]
    rows = []
    for row_name, values in zip(labels, matrix):
        row = {"gt\\pred": row_name}
        row.update({label: int(value) for label, value in zip(labels, values)})
        rows.append(row)
    write_csv(path, rows)


def class_name(cls):
    if cls == "all":
        return "all"
    if cls < len(CLASS_NAMES):
        return CLASS_NAMES[cls]
    return str(cls)


def print_table(rows):
    print(
        f"{'class':<12}{'tp':>8}{'fp':>8}{'fn':>8}"
        f"{'pred':>8}{'gt':>8}{'P':>10}{'R':>10}{'F1':>10}"
        f"{'MAE':>10}{'MRE':>10}"
    )
    for row in rows:
        print(
            f"{row['class']:<12}"
            f"{row['tp']:>8}{row['fp']:>8}{row['fn']:>8}"
            f"{row['pred_count']:>8}{row['gt_count']:>8}"
            f"{row['precision']:>10.4f}{row['recall']:>10.4f}{row['f1']:>10.4f}"
            f"{row['mae']:>10.2f}{row['mre']:>10.4f}"
        )


def main():
    pred_dir = Path(PRED_DIR)
    gt_dir = Path(GT_DIR)
    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"no gt labels found: {gt_dir}")

    stats = empty_stats()
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        pred = load_yolo_xywh(pred_path)
        gt = load_yolo_xywh(gt_path)
        update_stats(stats, pred, gt)

    rows = [metric_row(stats, cls) for cls in range(NUM_CLASSES)]
    rows.append(metric_row(stats, "all"))
    write_csv(OUT_CSV, rows)
    write_confusion(CONFUSION_CSV, stats["confusion"])

    all_row = rows[-1]
    print("pred dir:", PRED_DIR)
    print("gt dir:", GT_DIR)
    print("metrics csv:", OUT_CSV)
    print("confusion csv:", CONFUSION_CSV)
    print("iou:", IOU_THRESH)
    print(f"images: {all_row['num_images']}")
    print_table(rows)


if __name__ == "__main__":
    main()

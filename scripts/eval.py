"""评估单个 HeatmapYOLO 模型。"""

import csv
import os
from collections import defaultdict

import cv2

from config import CFG
from utils.infer_utils import (
    calculate_detection_metrics,
    infer_batch_small,
    load_heatmap_model,
    load_yolo_boxes,
)


CHECKPOINT_PATH = CFG.best_weight_path
PEAK_THRESH = 0.3
PEAK_KERNEL = 5
MATCH_IOU = 0.5
EVAL_BATCH_SIZE = 32
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_model(checkpoint_path=CHECKPOINT_PATH, num_classes=CFG.num_classes):
    return load_heatmap_model(weight_path=checkpoint_path, num_classes=num_classes)


def list_images(img_dir):
    return sorted(
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if name.lower().endswith(IMAGE_EXTS)
    )


def empty_stats():
    return {
        stride: {
            cls: defaultdict(float)
            for cls in list(range(CFG.num_classes)) + ["all"]
        }
        for stride in CFG.strides
    }


def update_stats(stats, stride, metrics):
    for cls, row in metrics.items():
        dst = stats[stride][cls]
        for key in ("tp", "fp", "fn", "pred_count", "gt_count", "count_error", "count_rel_error", "count_acc"):
            dst[key] += row[key]
        dst["images"] += 1


def finalize_row(row):
    tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    images = max(int(row["images"]), 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": row["count_error"] / images,
        "mre": row["count_rel_error"] / images,
        "count_acc": row["count_acc"] / images,
        "pred_count": int(row["pred_count"]),
        "gt_count": int(row["gt_count"]),
    }


def summarize_stats(stats):
    metrics = {"per_stride": {}}
    for stride in CFG.strides:
        stride_metrics = {"per_class": {}}
        for cls in range(CFG.num_classes):
            stride_metrics["per_class"][cls] = finalize_row(stats[stride][cls])
        stride_metrics["all"] = finalize_row(stats[stride]["all"])
        metrics["per_stride"][stride] = stride_metrics
    return metrics


def print_eval_table(metrics):
    print(
        f"\n{'stride':>8}{'class':>8}{'tp':>8}{'fp':>8}{'fn':>8}"
        f"{'precision':>12}{'recall':>12}{'f1':>10}{'mre':>10}{'cnt_acc':>10}"
    )
    for stride, stride_metrics in metrics["per_stride"].items():
        for cls, row in stride_metrics["per_class"].items():
            print(
                f"{stride:>8}{cls:>8}{row['tp']:>8}{row['fp']:>8}{row['fn']:>8}"
                f"{row['precision']:12.4f}{row['recall']:12.4f}{row['f1']:10.4f}"
                f"{row['mre']:10.4f}{row['count_acc']:10.4f}"
            )
        row = stride_metrics["all"]
        print(
            f"{stride:>8}{'all':>8}{row['tp']:>8}{row['fp']:>8}{row['fn']:>8}"
            f"{row['precision']:12.4f}{row['recall']:12.4f}{row['f1']:10.4f}"
            f"{row['mre']:10.4f}{row['count_acc']:10.4f}"
        )


def append_eval_csv(metrics, csv_path, epoch):
    if csv_path is None or epoch is None:
        return
    exists = os.path.exists(csv_path)
    legacy = False
    if exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        legacy = header == ["epoch", "stride", "class", "tp", "fp", "fn", "precision", "recall"]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "epoch",
                    "stride",
                    "class",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "mae",
                    "mre",
                    "count_acc",
                    "pred_count",
                    "gt_count",
                ]
            )
        for stride, stride_metrics in metrics["per_stride"].items():
            for cls, row in stride_metrics["per_class"].items():
                writer.writerow(_csv_row(epoch, stride, cls, row, legacy=legacy))
            writer.writerow(_csv_row(epoch, stride, "all", stride_metrics["all"], legacy=legacy))


def _csv_row(epoch, stride, cls, row, legacy=False):
    values = [
        epoch,
        stride,
        cls,
        row["tp"],
        row["fp"],
        row["fn"],
        row["precision"],
        row["recall"],
        row["f1"],
        row["mae"],
        row["mre"],
        row["count_acc"],
        row["pred_count"],
        row["gt_count"],
    ]
    return values[:8] if legacy else values


def eval(
    model=None,
    num_classes=CFG.num_classes,
    checkpoint_path=CHECKPOINT_PATH,
    test_root=CFG.test_root,
    print_table=True,
    csv_path=None,
    epoch=None,
    return_metrics=True,
    **_,
):
    if model is None:
        model = build_model(checkpoint_path=checkpoint_path, num_classes=num_classes)

    img_dir = os.path.join(test_root, "images")
    lab_dir = os.path.join(test_root, "labels")
    image_paths = list_images(img_dir)
    stats = empty_stats()

    for start in range(0, len(image_paths), EVAL_BATCH_SIZE):
        paths = image_paths[start:start + EVAL_BATCH_SIZE]
        images = []
        gt_batch = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            images.append(img)
            label_path = os.path.join(lab_dir, os.path.basename(path).rsplit(".", 1)[0] + ".txt")
            gt_batch.append(load_yolo_boxes(label_path, img.shape, num_classes=num_classes))

        pred_batch = infer_batch_small(
            model,
            images,
            batch_size=EVAL_BATCH_SIZE,
            strides=CFG.strides,
            decode_head="all",
            peak_thresh=PEAK_THRESH,
            peak_kernel=PEAK_KERNEL,
        )
        for pred_by_stride, gt_by_class in zip(pred_batch, gt_batch):
            for stride in CFG.strides:
                metrics = calculate_detection_metrics(
                    pred_by_stride[stride],
                    gt_by_class,
                    num_classes=num_classes,
                    iou_thresh=MATCH_IOU,
                )
                update_stats(stats, stride, metrics)

    metrics = summarize_stats(stats)
    if print_table:
        print_eval_table(metrics)
    append_eval_csv(metrics, csv_path, epoch)
    return metrics if return_metrics else None


if __name__ == "__main__":
    eval(print_table=True, return_metrics=False)

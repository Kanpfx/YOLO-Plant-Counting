"""Train an SSDLite MobileNetV3 non-YOLO detection baseline."""

from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import nms

from config import CFG


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
RUN_NAME = "ssdlite320_mobilenet_v3_large"
BATCH_SIZE = min(int(CFG.batch_size), 8)
WORKERS = min(4, os.cpu_count() or 2)
CONF_THRESH = 0.25
NMS_IOU = 0.5
MATCH_IOU = 0.5


def project_root() -> Path:
    return Path(getattr(CFG, "root", Path.cwd()))


def checkpoint_root() -> Path:
    path = Path(CFG.checkpoint_dir)
    path = path if path.is_absolute() else project_root() / path
    return path.parent if path.name.startswith("yolov") else path


OUT_DIR = checkpoint_root() / RUN_NAME


def dataset_root() -> Path:
    candidates = [Path(CFG.data_root), project_root() / "data" / "dataset"]
    for root in candidates:
        if (root / "train" / "images").is_dir():
            return root
    raise FileNotFoundError(f"Cannot find train/images under: {candidates}")


DATA_ROOT = dataset_root()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_AMP:
        torch.cuda.manual_seed_all(seed)


def load_yolo_label(path: Path, width: int, height: int):
    boxes, labels = [], []
    if not path.exists():
        return boxes, labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split()[:5])
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)
    return boxes, labels


class YoloDataset(Dataset):
    def __init__(self, root: Path, augment: bool):
        self.image_dir = root / "images"
        self.label_dir = root / "labels"
        self.augment = augment
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        boxes, labels = load_yolo_label(self.label_dir / f"{path.stem}.txt", w, h)

        if self.augment and boxes and random.random() < 0.5:
            img = img[:, ::-1].copy()
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}


def collate(batch):
    return tuple(zip(*batch))


def build_model():
    try:
        return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=CFG.num_classes + 1)
    except TypeError:
        return ssdlite320_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, num_classes=CFG.num_classes + 1)


def iou_matrix(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), np.float32)
    p, g = np.asarray(pred, np.float32), np.asarray(gt, np.float32)
    x1, y1 = np.maximum(p[:, None, 0], g[None, :, 0]), np.maximum(p[:, None, 1], g[None, :, 1])
    x2, y2 = np.minimum(p[:, None, 2], g[None, :, 2]), np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    return np.where(area_p[:, None] + area_g[None, :] - inter > 0, inter / (area_p[:, None] + area_g[None, :] - inter), 0)


def match(pred, gt):
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    pred = np.asarray(pred, np.float32)
    pred = pred[np.argsort(-pred[:, 4])]
    ious = iou_matrix(pred[:, :4], gt)
    used = np.zeros(len(gt), bool)
    tp = fp = 0
    for row in ious:
        row = row.copy()
        row[used] = -1
        j = int(row.argmax())
        if row[j] >= MATCH_IOU:
            used[j] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~used).sum())


def by_class_target(target):
    out = defaultdict(list)
    for box, label in zip(target["boxes"].cpu().numpy(), target["labels"].cpu().numpy()):
        out[int(label) - 1].append(tuple(float(x) for x in box))
    return out


def by_class_pred(pred):
    out = defaultdict(list)
    boxes = pred["boxes"].detach().cpu()
    scores = pred["scores"].detach().cpu()
    labels = pred["labels"].detach().cpu()
    keep = scores >= CONF_THRESH
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    for label in labels.unique():
        idx = torch.where(labels == label)[0]
        for j in idx[nms(boxes[idx], scores[idx], NMS_IOU)]:
            cls = int(labels[j]) - 1
            if 0 <= cls < CFG.num_classes:
                out[cls].append((*boxes[j].numpy().astype(float).tolist(), float(scores[j])))
    return out


def update_stats(stats, pred, gt):
    total_p = total_g = total_tp = total_fp = total_fn = 0
    for cls in range(CFG.num_classes):
        p, g = pred.get(cls, []), gt.get(cls, [])
        tp, fp, fn = match(p, g)
        stats[cls]["tp"] += tp; stats[cls]["fp"] += fp; stats[cls]["fn"] += fn
        stats[cls]["err"] += abs(len(p) - len(g)); stats[cls]["rel"] += abs(len(p) - len(g)) / max(len(g), 1)
        stats[cls]["acc"] += max(0.0, 1.0 - abs(len(p) - len(g)) / max(len(g), 1)); stats[cls]["n"] += 1
        total_p += len(p); total_g += len(g); total_tp += tp; total_fp += fp; total_fn += fn
    err = abs(total_p - total_g)
    stats["all"]["tp"] += total_tp; stats["all"]["fp"] += total_fp; stats["all"]["fn"] += total_fn
    stats["all"]["err"] += err; stats["all"]["rel"] += err / max(total_g, 1)
    stats["all"]["acc"] += max(0.0, 1.0 - err / max(total_g, 1)); stats["all"]["n"] += 1


def summarize(stats):
    rows = {}
    for cls, s in stats.items():
        tp, fp, fn, n = int(s["tp"]), int(s["fp"]), int(s["fn"]), max(int(s["n"]), 1)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        rows[cls] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
            "mae": s["err"] / n, "mre": s["rel"] / n, "count_acc": s["acc"] / n,
        }
    return rows


def train_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total = 0.0
    for step, (images, targets) in enumerate(loader, 1):
        images = [x.to(DEVICE, non_blocking=USE_AMP) for x in images]
        targets = [{k: v.to(DEVICE, non_blocking=USE_AMP) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            loss = sum(model(images, targets).values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        scaler.step(optimizer); scaler.update()
        total += float(loss.detach().cpu())
        if step % 20 == 0 or step == len(loader):
            print(f"epoch {epoch:03d} step {step:04d}/{len(loader):04d} loss={loss.item():.4f}")
    return total / max(len(loader), 1)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    stats = {cls: defaultdict(float) for cls in list(range(CFG.num_classes)) + ["all"]}
    for images, targets in loader:
        preds = model([x.to(DEVICE, non_blocking=USE_AMP) for x in images])
        for pred, target in zip(preds, targets):
            update_stats(stats, by_class_pred(pred), by_class_target(target))
    return summarize(stats)


def write_row(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(row[0])
        writer.writerow(row[1])


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "class_names": CFG.class_names}, path)


def main():
    seed_all(CFG.seed)
    train_loader = DataLoader(YoloDataset(DATA_ROOT / "train", True), batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, collate_fn=collate, pin_memory=USE_AMP)
    test_loader = DataLoader(YoloDataset(DATA_ROOT / "test", False), batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, collate_fn=collate, pin_memory=USE_AMP)
    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_mre, stale = float("inf"), 0

    for epoch in range(1, CFG.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, epoch)
        scheduler.step()
        save_model(model, OUT_DIR / "last.pt")
        write_row(OUT_DIR / "train_log.csv", (["epoch", "loss"], [epoch, loss]))

        if epoch % CFG.eval_interval and epoch != CFG.epochs:
            continue
        metrics = evaluate(model, test_loader)
        for cls, row in metrics.items():
            write_row(
                OUT_DIR / "eval_log.csv",
                (
                    ["epoch", "class", "tp", "fp", "fn", "precision", "recall", "f1", "mae", "mre", "count_acc"],
                    [epoch, cls, row["tp"], row["fp"], row["fn"], row["precision"], row["recall"], row["f1"], row["mae"], row["mre"], row["count_acc"]],
                ),
            )
        print(f"eval epoch {epoch:03d}: f1={metrics['all']['f1']:.4f} mre={metrics['all']['mre']:.4f} acc={metrics['all']['count_acc']:.4f}")
        if metrics["all"]["mre"] < best_mre:
            best_mre, stale = metrics["all"]["mre"], 0
            save_model(model, OUT_DIR / "best.pt")
        else:
            stale += 1
            if stale >= CFG.patience:
                print(f"early stop: best mre={best_mre:.4f}")
                break


if __name__ == "__main__":
    main()

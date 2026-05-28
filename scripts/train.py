"""训练三尺度 heatmap YOLO 模型。"""

import csv
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from config import CFG
from dataset import DetectionDataset
from eval import eval as run_eval
from loss import CenterNetLoss
from model import HeatmapYOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


class ModelEMA:
    """维护模型参数的指数滑动平均。"""

    def __init__(self, model, decay=0.999, warmup=200):
        self.ema = deepcopy(model).eval()
        self.decay, self.warmup, self.updates = decay, warmup, 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _decay(self):
        if self.warmup <= 0:
            return self.decay
        return self.decay * (1 - math.exp(-self.updates / self.warmup))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self._decay()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            mv = msd[k].detach()
            if torch.is_floating_point(v):
                v.mul_(d).add_(mv, alpha=1 - d)
            else:
                v.copy_(mv)

    def state_dict(self):
        return self.ema.state_dict()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_AMP:
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, ema, loader, criterion, optimizer, scheduler, scaler, epoch):
    """训练一个 epoch 并返回平均损失。"""
    model.train()
    sums = {"heat": 0.0, "center": 0.0, "wh": 0.0, "total": 0.0}
    steps = len(loader)

    for i, batch in enumerate(loader):
        imgs = batch["image"].to(DEVICE, non_blocking=USE_AMP)
        targets = batch["targets"]
        optimizer.zero_grad(set_to_none=True)

        # 三个尺度分别计算 loss，再按配置权重汇总。
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(imgs)
            loss = heat_l = center_l = wh_l = 0.0
            for pred, tgt, w in zip(preds, targets, CFG.loss_weights):
                if w == 0:
                    continue
                ph, pc, pw = model.split_output(pred)
                gh, gc, gw = [x.to(DEVICE, non_blocking=USE_AMP) for x in tgt]
                l, parts = criterion((ph, pc, pw), (gh, gc, gw))
                loss += w * l
                heat_l += w * parts["heatmap"]
                center_l += w * parts["center"]
                wh_l += w * criterion.wh_weight * parts["wh"]

        # AMP 发生 overflow 跳过 optimizer.step 时，不同步 EMA 和 scheduler。
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not USE_AMP or scaler.get_scale() >= before:
            scheduler.step()
            ema.update(model)

        sums["total"] += loss.item()
        sums["heat"] += heat_l.item()
        sums["center"] += center_l.item()
        sums["wh"] += wh_l.item()
        if i % 10 == 0 or i == steps - 1:
            mem = f"{torch.cuda.memory_reserved()/1e9:.2f}G" if USE_AMP else "0G"
            epoch_text = f"{epoch + 1}/{CFG.epochs}"
            iter_text = f"{i + 1}/{steps}"
            print(
                f"{epoch_text:>10}{iter_text:>9}{mem:>10}"
                f"{heat_l.item():14.4f}{center_l.item():13.4f}{wh_l.item():13.4f}{loss.item():13.4f}"
            )

    return {k: v / max(steps, 1) for k, v in sums.items()}


def main():
    """执行完整训练流程。"""
    seed_everything(CFG.seed)

    dataset = DetectionDataset(
        os.path.join(CFG.train_root, "images"),
        os.path.join(CFG.train_root, "labels"),
        num_classes=CFG.num_classes,
        augment=True,
    )
    workers = min(4, os.cpu_count() or 2)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=USE_AMP,
        persistent_workers=workers > 0,
    )

    yolo = YOLO(CFG.yolo_yaml).model
    model = HeatmapYOLO(yolo, num_classes=CFG.num_classes).to(DEVICE)
    ema = ModelEMA(model)
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    steps = len(loader)
    total_steps = CFG.epochs * steps

    def lr_lambda(step):
        if step < steps:
            return (step + 1) / max(steps, 1)
        progress = (step - steps) / max(total_steps - steps, 1)
        return 0.8 + 0.2 * (0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(CFG.checkpoint_dir, "train_log.csv")
    eval_log_path = os.path.join(CFG.checkpoint_dir, "eval_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "heatmap_loss", "center_loss", "wh_loss", "total_loss"])
    with open(eval_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "stride", "class", "tp", "fp", "fn", "precision", "recall"])

    best = float("inf")
    patience_counter = 0
    for epoch in range(CFG.epochs):
        print(f"\n{'Epoch':>10}{'Iter':>9}{'GPU_mem':>10}{'heatmap_loss':>14}{'center_loss':>13}{'wh_loss':>13}{'total_loss':>13}")
        losses = run_epoch(model, ema, loader, criterion, optimizer, scheduler, scaler, epoch)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, losses["heat"], losses["center"], losses["wh"], losses["total"]])

        torch.save(model.state_dict(), os.path.join(CFG.checkpoint_dir, "last.pt"))
        torch.save(ema.state_dict(), os.path.join(CFG.checkpoint_dir, "last_ema.pt"))
        if losses["total"] < best - 1e-2:
            best = losses["total"]
            patience_counter = 0
            torch.save(ema.state_dict(), os.path.join(CFG.checkpoint_dir, "best.pt"))
            print(f"\nNew best loss {best:.4f}")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{CFG.patience})")

        if (epoch + 1) % CFG.eval_interval == 0:
            run_eval(
                model=ema.ema,
                num_classes=CFG.num_classes,
                checkpoint_path=os.path.join(CFG.checkpoint_dir, "best.pt"),
                test_root=CFG.test_root,
                print_table=True,
                csv_path=eval_log_path,
                epoch=epoch + 1,
                return_metrics=False,
            )

        if patience_counter >= CFG.patience:
            print(f"\nEarly stopping. Best loss {best:.4f}")
            break


if __name__ == "__main__":
    main()

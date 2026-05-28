"""训练官方 YOLOv8n 对照模型。"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import yaml
from ultralytics import YOLO

from config import CFG


DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 4
RUN_NAME = "official_yolov8n"
PROJECT_ROOT = Path(CFG.root)
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


def work_path(path):
    """将相对路径固定到当前运行目录。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_data_yaml():
    """生成 Ultralytics 训练数据配置。"""
    data_yaml = work_path("yolov8_dataset.yaml")
    data = {
        "path": CFG.data_root,
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "nc": CFG.num_classes,
        "names": list(CFG.class_names),
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return data_yaml


def main():
    """启动官方 YOLOv8n 训练。"""
    data_yaml = write_data_yaml()

    model = YOLO("yolov8n.yaml")
    model.train(
        data=str(data_yaml),
        imgsz=CFG.patch_size,
        epochs=CFG.epochs,
        batch=CFG.batch_size,
        lr0=CFG.lr,
        weight_decay=CFG.weight_decay,
        optimizer="AdamW",
        patience=CFG.patience,
        seed=CFG.seed,
        device=DEVICE,
        workers=WORKERS,
        project=str(CHECKPOINT_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=False,
        amp=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    main()

"""论文实验配置。"""

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
OUTPUT_DIR = ROOT / "result"
TEST_ROOT = ROOT / "data/dataset/test"
os.environ.setdefault("YOLO_CONFIG_DIR", str(OUTPUT_DIR))

CLASS_NAMES = ("Maize", "Sugarbeet", "Rice")
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 1024
BATCH_SIZE = 16

IOU_THRESH = 0.5
HEATMAP_THRESH = 0.3
OFFICIAL_CONF = 0.25
GENERALIZATION_SCALES = (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5)

LATENCY_WARMUP = 10
LATENCY_RUNS = 50

RUN_PHYSICAL = True
RUN_PERFORMANCE = True
RUN_GENERALIZATION = True
SAVE_PLOTS = True

MODELS = {
    "ssdlite320_mobilenet_v3_large": {"name": "SSDLite-MobileNetV3", "color": "#64B5CD", "kind": "ssdlite"},
    "official_yolov8n": {"name": "YOLOv8n", "color": "#4C72B0", "kind": "official"},
    "yolov8n_p345": {"name": "P3/P4/P5 heatmap", "color": "#DD8452", "kind": "heatmap"},
    "yolov8n_p234": {"name": "P2/P3/P4 heatmap", "color": "#55A868", "kind": "heatmap"},
    "yolov8n_p234_high_loss_weight": {
        "name": "P234 high weight",
        "color": "#C44E52",
        "kind": "heatmap",
        "checkpoint_dir": "yolov8n_p234_loss_1_07_05",
    },
    "yolov8n_p234_equal_weight": {
        "name": "P234 equal weight",
        "color": "#937860",
        "kind": "heatmap",
        "checkpoint_dir": "yolov8n_p234_loss_1_1_1",
    },
    "yolov8n_p234_p2single": {"name": "P2 only", "color": "#8172B2", "kind": "heatmap"},
}

GROUPS = {
    "main": ("official_yolov8n", "yolov8n_p234", "yolov8n_p345", "ssdlite320_mobilenet_v3_large"),
    "ablation": ("official_yolov8n", "yolov8n_p234", "yolov8n_p345", "yolov8n_p234_p2single"),
    "generalization": ("official_yolov8n", "yolov8n_p345", "yolov8n_p234", "yolov8n_p234_p2single"),
}

VARIANTS = {
    "p234": {
        "yaml": "yolov8n_p234.yaml",
        "strides": (4, 8, 16),
        "feature_indices": (13, 16, 19),
        "feature_names": ("P2", "P3", "P4"),
    },
    "p345": {
        "yaml": "yolov8n_p345.yaml",
        "strides": (8, 16, 32),
        "feature_indices": (15, 18, 21),
        "feature_names": ("P3", "P4", "P5"),
    },
}


def parse_model_variant(checkpoint_dir):
    """读取训练时记录的 heatmap 模型变体。"""
    config_path = checkpoint_dir / "config.py"
    if not config_path.exists():
        return None
    match = re.search(r'model_variant\s*=\s*"([^"]+)"', config_path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1) if match else None


def build_model_list():
    """生成实验脚本使用的模型信息。"""
    models = []
    for dir_name, info in MODELS.items():
        kind = info["kind"]
        checkpoint_dir = ROOT / "checkpoints" / info.get("checkpoint_dir", dir_name)
        model = {
            "dir_name": dir_name,
            "name": info["name"],
            "color": info["color"],
            "kind": kind,
            "checkpoint_dir": checkpoint_dir,
            "variant": None,
            "yaml_path": None,
            "strides": None,
            "feature_indices": None,
            "feature_names": None,
        }

        if kind == "official":
            model["weight_path"] = checkpoint_dir / "weights" / "best.pt"
        elif kind == "ssdlite":
            model["weight_path"] = checkpoint_dir / "best.pt"
        else:
            variant = parse_model_variant(checkpoint_dir) or "p234"
            variant_info = VARIANTS[variant]
            model.update(
                {
                    "weight_path": checkpoint_dir / "best.pt",
                    "variant": variant,
                    "yaml_path": SCRIPT_DIR / variant_info["yaml"],
                    "strides": variant_info["strides"],
                    "feature_indices": variant_info["feature_indices"],
                    "feature_names": variant_info["feature_names"],
                }
            )

        if not model["weight_path"].exists():
            raise FileNotFoundError(f"缺少权重文件: {model['weight_path']}")
        models.append(model)
    return models


def select_models(models, group_name):
    by_name = {model["dir_name"]: model for model in models}
    return [by_name[name] for name in GROUPS[group_name] if name in by_name]

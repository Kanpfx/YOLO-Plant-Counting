"""训练和普通推理使用的配置。"""

from pathlib import Path


class CFG:
    root = Path(__file__).resolve().parents[1]
    script_dir = root / "scripts"
    data_root = str(root / "data" / "dataset")
    checkpoint_dir = str(root / "checkpoints" / "yolov8n_p234")
    input_image_path = str(root / "data" / "original" / "test" / "images" / "Maize (1).JPG")

    patch_size = 1024
    num_classes = 3
    class_names = ("Maize", "Sugarbeet", "Rice")

    model_variant = "p234"
    model_variants = {
        "p234": {
            "yaml": str(script_dir / "yolov8n_p234.yaml"),
            "strides": (4, 8, 16),
            "feature_indices": (13, 16, 19),
            "feature_names": ("P2", "P3", "P4"),
        },
        "p345": {
            "yaml": str(script_dir / "yolov8n_p345.yaml"),
            "strides": (8, 16, 32),
            "feature_indices": (15, 18, 21),
            "feature_names": ("P3", "P4", "P5"),
        },
    }

    batch_size = 24
    epochs = 50
    lr = 3e-4
    weight_decay = 1e-3
    grad_clip = 20.0
    loss_weights = (1.0, 0.3, 0.1)
    eval_interval = 5
    patience = 5
    seed = 114514

    if model_variant not in model_variants:
        raise ValueError(f"未知模型变体: {model_variant}")

    variant = model_variants[model_variant]
    yolo_yaml = variant["yaml"]
    strides = variant["strides"]
    feature_indices = variant["feature_indices"]
    feature_names = variant["feature_names"]

    train_root = str(Path(data_root) / "train")
    test_root = str(Path(data_root) / "test")
    best_weight_path = str(Path(checkpoint_dir) / "best.pt")
    onnx_output_path = str(Path(checkpoint_dir) / "p234_fp32_1024_b1.onnx")

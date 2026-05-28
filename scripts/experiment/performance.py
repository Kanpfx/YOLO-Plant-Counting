from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import nms

from common import (
    empty_stats,
    ensure_code_path,
    list_images,
    load_yolo_boxes,
    read_image,
    resolve_dataset_dirs,
    summarize_stats,
    update_stats,
    write_csv,
)
from experiment_config import IMG_SIZE, NUM_CLASSES
import experiment_config as exp_cfg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ensure_code_path()
from utils.infer_utils import infer_batch_small, load_heatmap_model
from utils.infer_utils import infer_sliding_window
from utils.infer_utils import make_window_coords, pad_patch


def configure_code_cfg(model_info):
    ensure_code_path()
    import config as train_config

    cfg = train_config.CFG
    cfg.patch_size = IMG_SIZE
    cfg.num_classes = NUM_CLASSES
    cfg.model_variant = model_info["variant"]
    cfg.yolo_yaml = str(model_info["yaml_path"])
    cfg.strides = model_info["strides"]
    cfg.feature_indices = model_info["feature_indices"]
    cfg.feature_names = model_info["feature_names"]
    return cfg


def build_heatmap_model(model_info):
    configure_code_cfg(model_info)
    return load_heatmap_model(
        weight_path=str(model_info["weight_path"]),
        yolo_yaml=str(model_info["yaml_path"]),
        num_classes=NUM_CLASSES,
        feature_indices=model_info["feature_indices"],
        device=DEVICE,
    )


def build_ssdlite_model(model_info):
    try:
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=NUM_CLASSES + 1)
    except TypeError:
        model = ssdlite320_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, num_classes=NUM_CLASSES + 1)
    try:
        state = torch.load(model_info["weight_path"], map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_info["weight_path"], map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def _needs_sliding(images, force_sliding=False):
    if force_sliding:
        return True
    return any(img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE for img in images)


def predict_heatmap_dataset(
    model_info,
    image_paths,
    batch_size=16,
    downsample=1,
    peak_thresh=0.3,
    sliding=False,
    restore_size=False,
    resize_scale=None,
):
    model = build_heatmap_model(model_info)
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs = [read_image(p, downsample=downsample, restore_size=restore_size, resize_scale=resize_scale) for p in batch_paths]
        if _needs_sliding(imgs, force_sliding=sliding):
            preds = []
            for img in imgs:
                boxes_by_stride, _ = infer_sliding_window(
                    model,
                    img,
                    batch_size=batch_size,
                    strides=model_info["strides"],
                    peak_thresh=peak_thresh,
                )
                preds.append(boxes_by_stride[0])
        else:
            preds = infer_batch_small(
                model,
                imgs,
                batch_size=batch_size,
                strides=model_info["strides"],
                device=DEVICE,
                decode_head="first",
                peak_thresh=peak_thresh,
            )
        for path, img, pred in zip(batch_paths, imgs, preds):
            yield path, img, pred


def _box_iou_one(box, boxes):
    if not boxes:
        return []
    x1 = torch.maximum(torch.tensor(box[0]), torch.tensor([b[0] for b in boxes]))
    y1 = torch.maximum(torch.tensor(box[1]), torch.tensor([b[1] for b in boxes]))
    x2 = torch.minimum(torch.tensor(box[2]), torch.tensor([b[2] for b in boxes]))
    y2 = torch.minimum(torch.tensor(box[3]), torch.tensor([b[3] for b in boxes]))
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    area_b = torch.tensor([max((b[2] - b[0]) * (b[3] - b[1]), 1e-6) for b in boxes])
    return (inter / (area_a + area_b - inter + 1e-6)).tolist()


def _nms_class_boxes(boxes, iou_thresh=0.5):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    for box in boxes:
        if all(iou <= iou_thresh for iou in _box_iou_one(box, keep)):
            keep.append(box)
    return keep


def _predict_official_image(model, img, conf=0.25, iou=0.7):
    result = model.predict(img, imgsz=IMG_SIZE, conf=conf, iou=iou, verbose=False, device=0 if DEVICE == "cuda" else "cpu")[0]
    by_class = defaultdict(list)
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, scores, classes):
            if 0 <= cls < NUM_CLASSES:
                by_class[cls].append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)))
    return by_class


def _predict_official_sliding(model, img, conf=0.25, iou=0.7):
    h, w = img.shape[:2]
    by_class = defaultdict(list)
    for x, y in make_window_coords(img.shape, patch_size=IMG_SIZE, overlap=0.1):
        raw_patch = img[y:y + IMG_SIZE, x:x + IMG_SIZE]
        valid_h, valid_w = raw_patch.shape[:2]
        patch = pad_patch(raw_patch, patch_size=IMG_SIZE)
        pred = _predict_official_image(model, patch, conf=conf, iou=iou)
        for cls, boxes in pred.items():
            for x1, y1, x2, y2, score in boxes:
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                if cx >= valid_w or cy >= valid_h:
                    continue
                by_class[cls].append(
                    (
                        max(0.0, min(w, x1 + x)),
                        max(0.0, min(h, y1 + y)),
                        max(0.0, min(w, x2 + x)),
                        max(0.0, min(h, y2 + y)),
                        score,
                    )
                )
    return {cls: _nms_class_boxes(boxes, iou_thresh=0.5) for cls, boxes in by_class.items()}


def predict_official_dataset(model_info, image_paths, downsample=1, conf=0.25, iou=0.7, sliding=False, restore_size=False, resize_scale=None):
    model = YOLO(str(model_info["weight_path"]))
    for path in image_paths:
        img = read_image(path, downsample=downsample, restore_size=restore_size, resize_scale=resize_scale)
        if sliding or img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            by_class = _predict_official_sliding(model, img, conf=conf, iou=iou)
        else:
            by_class = _predict_official_image(model, img, conf=conf, iou=iou)
        yield path, img, by_class


def _image_to_tensor(img):
    import cv2
    import numpy as np

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _predict_ssdlite_image(model, img, conf=0.25, iou=0.5):
    pred = model([_image_to_tensor(img).to(DEVICE)])[0]
    boxes = pred["boxes"].detach().cpu()
    scores = pred["scores"].detach().cpu()
    labels = pred["labels"].detach().cpu()
    keep = scores >= conf
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    by_class = defaultdict(list)
    for label in labels.unique():
        idx = torch.where(labels == label)[0]
        for j in idx[nms(boxes[idx], scores[idx], iou)]:
            cls = int(labels[j]) - 1
            if 0 <= cls < NUM_CLASSES:
                box = boxes[j].numpy().astype(float).tolist()
                by_class[cls].append((box[0], box[1], box[2], box[3], float(scores[j])))
    return by_class


def predict_ssdlite_dataset(model_info, image_paths, downsample=1, conf=0.25, iou=0.5, sliding=False, restore_size=False, resize_scale=None):
    if sliding:
        raise NotImplementedError("SSDLite 只用于固定尺寸小图评估。")
    model = build_ssdlite_model(model_info)
    for path in image_paths:
        img = read_image(path, downsample=downsample, restore_size=restore_size, resize_scale=resize_scale)
        yield path, img, _predict_ssdlite_image(model, img, conf=conf, iou=iou)


def evaluate_model(
    model_info,
    dataset_root,
    batch_size=16,
    downsample=1,
    iou_thresh=exp_cfg.IOU_THRESH,
    peak_thresh=exp_cfg.HEATMAP_THRESH,
    conf=exp_cfg.OFFICIAL_CONF,
    sliding=False,
    restore_size=False,
    resize_scale=None,
):
    image_dir, label_dir = resolve_dataset_dirs(Path(dataset_root))
    image_paths = list_images(image_dir)
    stats = empty_stats()

    if model_info["kind"] == "official":
        iterator = predict_official_dataset(
            model_info,
            image_paths,
            downsample=downsample,
            conf=conf,
            sliding=sliding,
            restore_size=restore_size,
            resize_scale=resize_scale,
        )
    elif model_info["kind"] == "ssdlite":
        iterator = predict_ssdlite_dataset(
            model_info,
            image_paths,
            downsample=downsample,
            conf=conf,
            iou=0.5,
            sliding=sliding,
            restore_size=restore_size,
            resize_scale=resize_scale,
        )
    else:
        iterator = predict_heatmap_dataset(
            model_info,
            image_paths,
            batch_size=batch_size,
            downsample=downsample,
            peak_thresh=peak_thresh,
            sliding=sliding,
            restore_size=restore_size,
            resize_scale=resize_scale,
        )

    for path, img, pred_by_class in iterator:
        h, w = img.shape[:2]
        gt_by_class = load_yolo_boxes(label_dir / f"{path.stem}.txt", width=w, height=h)
        update_stats(stats, pred_by_class, gt_by_class, iou_thresh=iou_thresh)

    if model_info["kind"] == "official":
        head = "official"
    elif model_info["kind"] == "ssdlite":
        head = "ssdlite"
    else:
        head = model_info["feature_names"][0]
    return summarize_stats(
        stats,
        model=model_info["dir_name"],
        display_name=model_info["name"],
        extra={"downsample": downsample, "resize_scale": resize_scale or 1.0, "restore_size": restore_size, "head": head},
    )


def run_performance(models, dataset_root, out_csv, batch_size=16):
    rows = []
    for model_info in models:
        rows.extend(evaluate_model(model_info, dataset_root, batch_size=batch_size))

    write_csv(Path(out_csv), rows)
    return rows

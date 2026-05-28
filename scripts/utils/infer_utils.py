"""推理、解码、可视化和指标计算工具。"""

from __future__ import annotations

import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from config import CFG
from model import HeatmapYOLO


def resolve_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device, enabled=True):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=(enabled and device == "cuda"))
    return torch.cuda.amp.autocast(enabled=(enabled and device == "cuda"))


def load_heatmap_model(
    weight_path=None,
    yolo_yaml=None,
    num_classes=None,
    feature_indices=None,
    device=None,
):
    """Build a HeatmapYOLO model and load a state dict."""
    device = resolve_device(device)
    weight_path = weight_path or CFG.best_weight_path
    yolo_yaml = yolo_yaml or CFG.yolo_yaml
    num_classes = num_classes or CFG.num_classes
    feature_indices = feature_indices or CFG.feature_indices

    yolo = YOLO(yolo_yaml).model
    model = HeatmapYOLO(yolo, num_classes=num_classes, feature_indices=feature_indices).to(device)
    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = _clean_state_dict(state)
    load_result = model.load_state_dict(state, strict=False)
    _check_load_result(load_result)
    model.eval()
    return model


def _clean_state_dict(state):
    """Remove non-parameter buffers added by profiling tools such as THOP."""
    drop_suffixes = ("total_ops", "total_params")
    return {
        key: value
        for key, value in state.items()
        if not key.endswith(drop_suffixes)
    }


def _check_load_result(load_result):
    """Allow profiling buffers to be missing/unexpected, fail on real mismatch."""
    allowed_suffixes = ("total_ops", "total_params")
    missing = [
        key for key in load_result.missing_keys
        if not key.endswith(allowed_suffixes)
    ]
    unexpected = [
        key for key in load_result.unexpected_keys
        if not key.endswith(allowed_suffixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load HeatmapYOLO weights. "
            f"missing={missing}, unexpected={unexpected}"
        )


def preprocess_bgr_images(images):
    """Convert a list of BGR uint8 images to an RGB float tensor."""
    tensors = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors, dim=0)


def load_yolo_boxes(label_path, image_shape, num_classes=None):
    """Read YOLO labels and return absolute xyxy boxes grouped by class."""
    num_classes = num_classes or CFG.num_classes
    h, w = image_shape[:2]
    boxes = defaultdict(list)
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = map(float, line.split()[:5])
            cls = int(cls)
            if not (0 <= cls < num_classes):
                continue
            x1 = (cx - bw * 0.5) * w
            y1 = (cy - bh * 0.5) * h
            x2 = (cx + bw * 0.5) * w
            y2 = (cy + bh * 0.5) * h
            boxes[cls].append((x1, y1, x2, y2))
    return boxes


def merge_close_points(candidates, min_center):
    """Suppress duplicate center responses with a lightweight grid search."""
    if len(candidates) == 0:
        return []
    dist2 = float(min_center * min_center)
    grid, keep = {}, []
    for c in candidates:
        ccx, ccy = float(c[5]), float(c[6])
        gx, gy = int(ccx // min_center), int(ccy // min_center)
        duplicate = False
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for kx, ky in grid.get((nx, ny), []):
                    if (ccx - kx) ** 2 + (ccy - ky) ** 2 < dist2:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        keep.append((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])))
        grid.setdefault((gx, gy), []).append((ccx, ccy))
    return keep


def decode_outputs_batch(
    model,
    outputs,
    strides=None,
    peak_thresh=0.3,
    peak_kernel=5,
    min_center=12,
):
    """Decode model outputs with local peaks and center-distance suppression."""
    strides = strides or CFG.strides
    batch_size = outputs[0].shape[0]
    predictions = [{stride: defaultdict(list) for stride in strides} for _ in range(batch_size)]

    for stride, packed in zip(strides, outputs):
        heatmap, center, wh = model.split_output(packed)
        heatmap = torch.sigmoid(heatmap)
        pooled = F.max_pool2d(heatmap, peak_kernel, stride=1, padding=peak_kernel // 2)
        peaks = (heatmap == pooled) & (heatmap > peak_thresh)

        for b in range(batch_size):
            for cls in range(heatmap.shape[1]):
                ys, xs = torch.where(peaks[b, cls])
                if ys.numel() == 0:
                    continue

                scores = heatmap[b, cls, ys, xs]
                dx = center[b, 0, ys, xs]
                dy = center[b, 1, ys, xs]
                bw = wh[b, 0, ys, xs] * stride
                bh = wh[b, 1, ys, xs] * stride
                cx = (xs.float() + dx) * stride
                cy = (ys.float() + dy) * stride

                cand = torch.stack(
                    (cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5, scores, cx, cy),
                    dim=1,
                )
                img_h = heatmap.shape[2] * stride
                img_w = heatmap.shape[3] * stride
                cand[:, 0] = cand[:, 0].clamp(0, img_w)
                cand[:, 1] = cand[:, 1].clamp(0, img_h)
                cand[:, 2] = cand[:, 2].clamp(0, img_w)
                cand[:, 3] = cand[:, 3].clamp(0, img_h)
                cand = cand[torch.argsort(cand[:, 4], descending=True)].detach().cpu().numpy()
                predictions[b][stride][cls] = merge_close_points(cand, min_center)
    return predictions


@torch.inference_mode()
def infer_batch_small(
    model,
    images,
    batch_size=16,
    strides=None,
    device=None,
    decode_head="first",
    peak_thresh=0.3,
    peak_kernel=5,
):
    """Run batched inference on fixed-size images.

    If decode_head is "all", each item is {stride: {class: boxes}}.
    Otherwise each item is {class: boxes} using the first stride.
    """
    device = resolve_device(device)
    strides = strides or CFG.strides
    use_amp = device == "cuda"
    decoded = []

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        batch = preprocess_bgr_images(batch_imgs).to(device, non_blocking=use_amp)
        with autocast_context(device, enabled=use_amp):
            outputs = model(batch)
        batch_pred = decode_outputs_batch(
            model,
            outputs,
            strides=strides,
            peak_thresh=peak_thresh,
            peak_kernel=peak_kernel,
        )
        if decode_head == "all":
            decoded.extend(batch_pred)
        else:
            decoded.extend(pred[strides[0]] for pred in batch_pred)
    return decoded


def sliding_positions(length, patch_size=None, overlap=0.1):
    patch_size = patch_size or CFG.patch_size
    step = int(patch_size * (1 - overlap))
    positions = list(range(0, max(1, length - patch_size + 1), step))
    end = max(0, length - patch_size)
    if not positions or positions[-1] != end:
        positions.append(end)
    return positions


def make_window_coords(image_shape, patch_size=None, overlap=0.1):
    patch_size = patch_size or CFG.patch_size
    h, w = image_shape[:2]
    xs = sliding_positions(w, patch_size=patch_size, overlap=overlap)
    ys = sliding_positions(h, patch_size=patch_size, overlap=overlap)
    return [(x, y) for y in ys for x in xs]


def pad_patch(patch, patch_size=None, value=114):
    """Pad boundary windows to the fixed network input size."""
    patch_size = patch_size or CFG.patch_size
    h, w = patch.shape[:2]
    if h == patch_size and w == patch_size:
        return patch
    padded = np.full((patch_size, patch_size, patch.shape[2]), value, dtype=patch.dtype)
    padded[:h, :w] = patch
    return padded


def init_fusion_levels(image_shape, strides=None, num_classes=None, device=None):
    h, w = image_shape[:2]
    strides = strides or CFG.strides
    num_classes = num_classes or CFG.num_classes
    device = resolve_device(device)
    levels = []
    for stride in strides:
        fh = max(1, h // stride)
        fw = max(1, w // stride)
        levels.append(
            {
                "hm": torch.zeros((num_classes, fh, fw), dtype=torch.float32, device=device),
                "ctr": torch.zeros((2, fh, fw), dtype=torch.float32, device=device),
                "wh": torch.zeros((2, fh, fw), dtype=torch.float32, device=device),
                "count": torch.zeros((fh, fw), dtype=torch.float32, device=device),
            }
        )
    return levels


def fuse_window_outputs(model, levels, outputs, coords, image_shape, strides=None, patch_size=None):
    strides = strides or CFG.strides
    patch_size = patch_size or CFG.patch_size
    for i, stride in enumerate(strides):
        hm, ctr, wh = model.split_output(outputs[i])
        hm = torch.sigmoid(hm)
        level = levels[i]

        for b, (x, y) in enumerate(coords):
            y0 = y // stride
            x0 = x // stride
            valid_h = level["hm"].shape[1] - y0
            valid_w = level["hm"].shape[2] - x0
            if valid_h <= 0 or valid_w <= 0:
                continue
            hm_sub = hm[b, :, :valid_h, :valid_w]
            ctr_sub = ctr[b, :, :valid_h, :valid_w]
            wh_sub = wh[b, :, :valid_h, :valid_w]
            ysl = slice(y0, y0 + hm_sub.shape[1])
            xsl = slice(x0, x0 + hm_sub.shape[2])

            level["hm"][:, ysl, xsl] += hm_sub
            level["ctr"][:, ysl, xsl] += ctr_sub
            level["wh"][:, ysl, xsl] += wh_sub
            level["count"][ysl, xsl] += 1


def average_fusion_levels(levels):
    for level in levels:
        count = level["count"].clamp_min(1.0)
        level["hm"] = level["hm"] / count.unsqueeze(0)
        level["ctr"] = level["ctr"] / count.unsqueeze(0)
        level["wh"] = level["wh"] / count.unsqueeze(0)


def gaussian_kernel2d(sigma, device):
    if sigma <= 0:
        return None
    radius = max(1, int(3 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (kernel_1d[:, None] @ kernel_1d[None, :]).float()


def smooth_heatmap(heatmap, sigma):
    kernel = gaussian_kernel2d(sigma, heatmap.device)
    if kernel is None:
        return heatmap
    channels = heatmap.shape[0]
    k = kernel.shape[0]
    weight = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)
    return F.conv2d(heatmap.unsqueeze(0), weight, padding=k // 2, groups=channels).squeeze(0)


def merge_box_centers(candidates, merge_factor=0.5):
    if len(candidates) == 0:
        return []
    keep, grid = [], {}
    for c in candidates:
        x1, y1, x2, y2, score, cx, cy = c.tolist()
        side = max(min(x2 - x1, y2 - y1), 1.0)
        gx, gy = int(cx // side), int(cy // side)
        merged = False
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for idx in grid.get((nx, ny), []):
                    old = keep[idx]
                    old_side = max(min(old[2] - old[0], old[3] - old[1]), 1.0)
                    threshold = merge_factor * min(side, old_side)
                    if (cx - old[5]) ** 2 + (cy - old[6]) ** 2 <= threshold * threshold:
                        total = old[4] + score + 1e-9
                        old[0] = (old[0] * old[4] + x1 * score) / total
                        old[1] = (old[1] * old[4] + y1 * score) / total
                        old[2] = (old[2] * old[4] + x2 * score) / total
                        old[3] = (old[3] * old[4] + y2 * score) / total
                        old[5] = (old[5] * old[4] + cx * score) / total
                        old[6] = (old[6] * old[4] + cy * score) / total
                        old[4] = max(old[4], score)
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            idx = len(keep)
            keep.append([x1, y1, x2, y2, score, cx, cy])
            grid.setdefault((gx, gy), []).append(idx)
    return keep


def decode_fused_level(
    level,
    stride,
    image_shape,
    peak_thresh=0.3,
    peak_kernel=5,
    gauss_sigma=1.0,
    merge_factor=0.5,
    min_box_side=5.0,
    min_center=12,
):
    h, w = image_shape[:2]
    heatmap = level["hm"]
    ctr, wh = level["ctr"], level["wh"]
    boxes = defaultdict(list)
    kernel = peak_kernel if peak_kernel % 2 == 1 else peak_kernel + 1

    for cls in range(heatmap.shape[0]):
        cls_hm = heatmap[cls:cls + 1].unsqueeze(0)
        pooled = F.max_pool2d(cls_hm, kernel, stride=1, padding=kernel // 2)
        ys, xs = torch.where((cls_hm == pooled) & (cls_hm > peak_thresh))[2:]
        if ys.numel() == 0:
            continue
        scores = cls_hm[0, 0, ys, xs]
        dx, dy = ctr[0, ys, xs], ctr[1, ys, xs]
        bw = wh[0, ys, xs] * stride
        bh = wh[1, ys, xs] * stride
        cx = (xs.float() + dx) * stride
        cy = (ys.float() + dy) * stride
        cand = torch.stack(
            (cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5, scores, cx, cy),
            dim=1,
        )
        cand = cand[torch.argsort(cand[:, 4], descending=True)].detach().cpu().numpy()
        for x1, y1, x2, y2, score in merge_close_points(cand, min_center=min_center):
            boxes[cls].append(
                (
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(w, int(x2)),
                    min(h, int(y2)),
                    float(score),
                )
            )
    return boxes


@torch.inference_mode()
def infer_sliding_window(
    model,
    image,
    batch_size=16,
    strides=None,
    patch_size=None,
    overlap=0.05,
    edge_discard=0.0,
    device=None,
    peak_thresh=0.3,
    peak_kernel=5,
):
    """Run sliding-window inference with average heatmap/offset/size fusion."""
    device = resolve_device(device)
    strides = strides or CFG.strides
    patch_size = patch_size or CFG.patch_size
    use_amp = device == "cuda"
    coords = make_window_coords(image.shape, patch_size=patch_size, overlap=overlap)
    levels = init_fusion_levels(image.shape, strides=strides, device=device)

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        patches = [pad_patch(image[y:y + patch_size, x:x + patch_size], patch_size=patch_size) for x, y in batch_coords]
        batch = preprocess_bgr_images(patches).to(device, non_blocking=use_amp)
        with autocast_context(device, enabled=use_amp):
            outputs = model(batch)
        fuse_window_outputs(model, levels, outputs, batch_coords, image.shape, strides, patch_size)

    average_fusion_levels(levels)
    boxes_by_stride = []
    for stride, level in zip(strides, levels):
        boxes_by_stride.append(
            decode_fused_level(
                level,
                stride=stride,
                image_shape=image.shape,
                peak_thresh=peak_thresh,
                peak_kernel=peak_kernel,
            )
        )
    return boxes_by_stride, levels


def box_iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    pred = np.asarray(pred_boxes, dtype=np.float32)
    gt = np.asarray(gt_boxes, dtype=np.float32)
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_p = np.clip(pred[:, 2] - pred[:, 0], 0.0, None) * np.clip(pred[:, 3] - pred[:, 1], 0.0, None)
    area_g = np.clip(gt[:, 2] - gt[:, 0], 0.0, None) * np.clip(gt[:, 3] - gt[:, 1], 0.0, None)
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def match_boxes(pred, gt, iou_thresh=0.5):
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    pred = np.asarray(pred, dtype=np.float32)
    pred = pred[np.argsort(-pred[:, 4])]
    gt = np.asarray(gt, dtype=np.float32)
    iou = box_iou_matrix(pred[:, :4], gt)
    matched = np.zeros(len(gt), dtype=bool)
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


def calculate_detection_metrics(pred_by_class, gt_by_class, num_classes=None, iou_thresh=0.5):
    """Compare prediction and ground-truth arrays for one image."""
    num_classes = num_classes or CFG.num_classes
    rows = {}
    total_tp = total_fp = total_fn = 0
    total_pred = total_gt = 0
    for cls in range(num_classes):
        pred = pred_by_class.get(cls, [])
        gt = gt_by_class.get(cls, [])
        tp, fp, fn = match_boxes(pred, gt, iou_thresh=iou_thresh)
        rows[cls] = _metric_row(tp, fp, fn, len(pred), len(gt))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pred += len(pred)
        total_gt += len(gt)
    rows["all"] = _metric_row(total_tp, total_fp, total_fn, total_pred, total_gt)
    return rows


def _metric_row(tp, fp, fn, pred_count, gt_count):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    count_error = abs(pred_count - gt_count)
    rel_error = count_error / max(gt_count, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "count_error": count_error,
        "count_rel_error": rel_error,
        "count_acc": max(0.0, 1.0 - rel_error),
    }


def save_visualizations(image, levels, boxes_by_stride, out_dir, strides=None, jpeg_quality=85):
    os.makedirs(out_dir, exist_ok=True)
    strides = strides or CFG.strides
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 180, 0)]
    h, w = image.shape[:2]
    for i, stride in enumerate(strides):
        vis = image.copy()
        for cls, boxes in boxes_by_stride[i].items():
            color = colors[cls % len(colors)]
            for x1, y1, x2, y2, score in boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{score:.2f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(os.path.join(out_dir, f"bbox_stride{stride}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])

        hm = levels[i]["hm"].detach().cpu().numpy()
        cls_id = int(np.argmax(hm.mean(axis=(1, 2))))
        heat = hm[cls_id]
        heat = (heat / (heat.max() + 1e-6) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (w, h))
        overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
        cv2.imwrite(os.path.join(out_dir, f"heatmap_overlay_stride{stride}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])

"""读取检测数据并生成多尺度 heatmap 标签。"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG

GAUSSIAN_CACHE = {}


def get_gaussian(radius):
    if radius not in GAUSSIAN_CACHE:
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        sigma = (2 * radius + 1) / 5
        GAUSSIAN_CACHE[radius] = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
    return GAUSSIAN_CACHE[radius]


def draw_gaussian(heatmap, x, y, radius):
    """在 heatmap 指定位置叠加高斯响应。"""
    x, y = int(x), int(y)
    h, w = heatmap.shape
    gaussian = get_gaussian(radius)

    left, right = max(0, x - radius), min(w, x + radius + 1)
    top, bottom = max(0, y - radius), min(h, y + radius + 1)
    g_left = left - (x - radius)
    g_right = g_left + (right - left)
    g_top = top - (y - radius)
    g_bottom = g_top + (bottom - top)

    if left < right and top < bottom:
        heatmap[top:bottom, left:right] = np.maximum(
            heatmap[top:bottom, left:right],
            gaussian[g_top:g_bottom, g_left:g_right],
        )


def load_boxes(label_path):
    """读取 YOLO 格式标签。"""
    if not os.path.exists(label_path):
        return np.zeros((0, 5), np.float32)
    with open(label_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if not rows:
        return np.zeros((0, 5), np.float32)
    return np.array([list(map(float, row.split())) for row in rows], np.float32)


class Augment:
    """对图像和 bbox 做同步增强。"""

    def __init__(self, flip=0.5, rotate=10, scale=0.2, translate=0.1, hsv=0.05):
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        self.trans = translate
        self.hsv = hsv

    def __call__(self, img, boxes):
        if len(boxes) == 0:
            return img, boxes

        boxes = boxes.copy()
        cx, cy = boxes[:, 1:3].T * CFG.patch_size
        w, h = boxes[:, 3:5].T * CFG.patch_size

        if random.random() < self.flip:
            img = img[:, ::-1]
            cx = CFG.patch_size - cx
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5

        # 仿射变换后，用四角点重算外接框。
        angle = random.uniform(-self.rotate, self.rotate)
        scale = 1 + random.uniform(-self.scale, self.scale)
        tx = random.uniform(-self.trans, self.trans) * CFG.patch_size
        ty = random.uniform(-self.trans, self.trans) * CFG.patch_size
        transform = cv2.getRotationMatrix2D((CFG.patch_size / 2, CFG.patch_size / 2), angle, scale)
        transform[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, transform, (CFG.patch_size, CFG.patch_size), borderValue=(114, 114, 114))

        corners = np.stack(
            [
                np.stack([x1, y1], 1),
                np.stack([x2, y1], 1),
                np.stack([x2, y2], 1),
                np.stack([x1, y2], 1),
            ],
            axis=1,
        )
        flat = corners.reshape(-1, 2)
        flat_h = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)], axis=1)
        warped = (transform @ flat_h.T).T.reshape(-1, 4, 2)

        x_min = np.clip(warped[:, :, 0].min(axis=1), 0, CFG.patch_size)
        y_min = np.clip(warped[:, :, 1].min(axis=1), 0, CFG.patch_size)
        x_max = np.clip(warped[:, :, 0].max(axis=1), 0, CFG.patch_size)
        y_max = np.clip(warped[:, :, 1].max(axis=1), 0, CFG.patch_size)

        w_new = x_max - x_min
        h_new = y_max - y_min
        valid = (w_new >= 2.0) & (h_new >= 2.0)
        if not np.any(valid):
            return img, np.zeros((0, 5), np.float32)

        cls = boxes[:, 0][valid]
        cx_new = (x_min[valid] + x_max[valid]) * 0.5
        cy_new = (y_min[valid] + y_max[valid]) * 0.5
        boxes = np.stack([cls, cx_new, cy_new, w_new[valid], h_new[valid]], axis=1).astype(np.float32)
        boxes[:, 1:5] /= CFG.patch_size

        if self.hsv > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 1 + random.uniform(-self.hsv, self.hsv)
            hsv[..., 2] *= 1 + random.uniform(-self.hsv, self.hsv)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img, boxes


class DetectionDataset(Dataset):
    """读取 patch 数据并生成三个尺度的监督信号。"""

    def __init__(self, img_dir, lbl_dir, num_classes=1, augment=True):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.num_classes = num_classes
        self.files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        self.img_paths = [os.path.join(self.img_dir, f) for f in self.files]
        self.lbl_paths = [os.path.join(self.lbl_dir, f.replace(".jpg", ".txt")) for f in self.files]
        self.empty_boxes = np.zeros((0, 5), np.float32)

        self.label_cache = {name: load_boxes(lbl_path) for name, lbl_path in zip(self.files, self.lbl_paths)}
        self.stride_meta = {
            stride: {
                "size": CFG.patch_size // stride,
                "px_scale": CFG.patch_size / stride,
                "radius_cap": int(64 / stride),
            }
            for stride in CFG.strides
        }
        self.target_templates = {
            stride: (
                np.zeros((self.num_classes, meta["size"], meta["size"]), np.float32),
                np.zeros((2, meta["size"], meta["size"]), np.float32),
                np.zeros((2, meta["size"], meta["size"]), np.float32),
            )
            for stride, meta in self.stride_meta.items()
        }
        self.aug = Augment() if augment else None

    def __len__(self):
        return len(self.files)

    def build_targets(self, boxes, stride):
        """为指定 stride 生成 heatmap、中心偏移和宽高标签。"""
        meta = self.stride_meta[stride]
        size = meta["size"]
        scale = meta["px_scale"]
        radius_cap = meta["radius_cap"]

        base_hm, base_ctr, base_wh = self.target_templates[stride]
        heatmap = base_hm.copy()
        center = base_ctr.copy()
        wh = base_wh.copy()

        if len(boxes) == 0:
            return heatmap, center, wh

        # 每个目标只在中心点写回归标签，在类别 heatmap 上绘制高斯峰。
        for cls, cx, cy, w, h in boxes:
            cls = int(cls)
            if not (0 <= cls < self.num_classes):
                continue
            x, y = cx * scale, cy * scale
            ix, iy = int(x), int(y)
            if not (0 <= ix < size and 0 <= iy < size):
                continue

            bw, bh = w * scale, h * scale
            bw_img, bh_img = w * CFG.patch_size, h * CFG.patch_size
            radius = min(int(max(1, min(bw_img, bh_img) / (2 * stride))), radius_cap)
            draw_gaussian(heatmap[cls], x, y, radius)
            center[:, iy, ix] = (x - ix, y - iy)
            wh[:, iy, ix] = (bw, bh)
        return heatmap, center, wh

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self.label_cache.get(name, self.empty_boxes)
        if self.aug and len(boxes) > 0:
            img, boxes = self.aug(img, boxes)

        targets = [
            tuple(torch.from_numpy(x) for x in self.build_targets(boxes, stride))
            for stride in CFG.strides
        ]
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        img *= 1.0 / 255.0
        return {"image": torch.from_numpy(img), "targets": targets}


if __name__ == "__main__":
    dataset = DetectionDataset(
        os.path.join(CFG.train_root, "images"),
        os.path.join(CFG.train_root, "labels"),
        num_classes=CFG.num_classes,
        augment=True,
    )
    print("Dataset size:", len(dataset))

    data = dataset[0]
    for i, (hm, ctr, wh) in enumerate(data["targets"]):
        print(f"\nstride {CFG.strides[i]}")
        print("heatmap", hm.shape)
        print("center", ctr.shape)
        print("size", wh.shape)

    import matplotlib.pyplot as plt

    img = data["image"].numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("image")
    plt.axis("off")
    for i, (hm, _, _) in enumerate(data["targets"]):
        plt.subplot(1, 4, i + 2)
        plt.imshow(hm[1].numpy(), cmap="jet")
        plt.title(f"heatmap stride {CFG.strides[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

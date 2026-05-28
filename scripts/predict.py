"""Run sliding-window inference on a high-resolution image."""

import cv2

from config import CFG
from utils.infer_utils import infer_sliding_window, load_heatmap_model, save_visualizations


OVERLAP = 0.05
EDGE_DISCARD = 0.0
INFER_BATCH_SIZE = 16
PEAK_THRESH = 0.3
PEAK_KERNEL = 5
SAVE_VIS = False
VIS_OUT_DIR = "."
JPEG_QUALITY = 85


def print_counts(boxes_by_stride):
    for stride, boxes_by_class in zip(CFG.strides, boxes_by_stride):
        total = sum(len(v) for v in boxes_by_class.values())
        detail = ", ".join(
            f"{CFG.class_names[cls]}={len(boxes_by_class.get(cls, []))}"
            for cls in range(CFG.num_classes)
        )
        print(f"Stride {stride}: {total} ({detail})")


def main():
    model = load_heatmap_model(CFG.best_weight_path)
    image = cv2.imread(CFG.input_image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {CFG.input_image_path}")
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    boxes_by_stride, levels = infer_sliding_window(
        model,
        image,
        batch_size=INFER_BATCH_SIZE,
        strides=CFG.strides,
        patch_size=CFG.patch_size,
        overlap=OVERLAP,
        edge_discard=EDGE_DISCARD,
        peak_thresh=PEAK_THRESH,
        peak_kernel=PEAK_KERNEL,
    )
    print_counts(boxes_by_stride)

    if SAVE_VIS:
        save_visualizations(
            image,
            levels,
            boxes_by_stride,
            VIS_OUT_DIR,
            strides=CFG.strides,
            jpeg_quality=JPEG_QUALITY,
        )


if __name__ == "__main__":
    main()

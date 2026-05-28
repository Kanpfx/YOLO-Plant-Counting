import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import onnx
import torch
from ultralytics import YOLO
from onnxsim import simplify
import onnxoptimizer

from model import HeatmapYOLO
from config import CFG

WEIGHT_PATH = CFG.best_weight_path
ONNX_PATH = CFG.onnx_output_path
BRANCH_INDEX = 0
BATCH_SIZE = 1
PATCH_SIZE = 1024
NUM_CLASSES = 3
OPSET_VERSION = 13
INPUT_NAME = "images"


class DeployBranch(torch.nn.Module):
    """只保留部署需要的一个 heatmap 分支。"""

    def __init__(self, model):
        super().__init__()
        self.layers = model.backbone.layers
        self.feature_index = model.feature_indices[BRANCH_INDEX]
        self.reduce = (model.reduce.p2, model.reduce.p3, model.reduce.p4)[BRANCH_INDEX]
        self.head = (model.head_p2, model.head_p3, model.head_p4)[BRANCH_INDEX]

    def forward(self, x):
        outs, feat = [], x
        for i, layer in enumerate(self.layers):
            if i > self.feature_index:
                break
            # YOLOv8 层间有跳连，按原 layer.f 关系逐层取输入。
            feat = outs[layer.f] if isinstance(layer.f, int) and layer.f != -1 else feat
            if not isinstance(layer.f, int):
                feat = [feat if j == -1 else outs[j] for j in layer.f]
            feat = layer(feat)
            outs.append(feat)
        return self.head(self.reduce(outs[self.feature_index]))


def main():
    CFG.patch_size = PATCH_SIZE
    CFG.num_classes = NUM_CLASSES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = CFG.strides[BRANCH_INDEX]
    output_name = f"{CFG.feature_names[BRANCH_INDEX].lower()}_head"
    input_shape = [BATCH_SIZE, 3, PATCH_SIZE, PATCH_SIZE]
    output_shape = [BATCH_SIZE, NUM_CLASSES + 4, PATCH_SIZE // stride, PATCH_SIZE // stride]
    raw_path = ONNX_PATH.replace(".onnx", "_raw.onnx")

    base = HeatmapYOLO(
        YOLO(CFG.yolo_yaml).model,
        num_classes=NUM_CLASSES,
        feature_indices=CFG.feature_indices,
    ).to(device)

    # 去掉 thop 等统计工具写入的非模型参数。
    state = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    state = {k: v for k, v in state.items() if not k.endswith(("total_ops", "total_params"))}

    result = base.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith(("total_ops", "total_params"))]
    unexpected = [k for k in result.unexpected_keys if not k.endswith(("total_ops", "total_params"))]
    if missing or unexpected:
        raise RuntimeError(f"权重加载失败: missing={missing}, unexpected={unexpected}")

    model = DeployBranch(base).to(device).eval()
    dummy = torch.zeros(input_shape, device=device)
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            raw_path,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[output_name],
            dynamic_axes=None,
            keep_initializers_as_inputs=False,
        )

    onnx_model = onnx.load(raw_path)
    onnx_model.graph.input[0].name = INPUT_NAME
    onnx_model.graph.output[0].name = output_name

    # 固定输入输出 shape，便于 TensorRT 按静态尺寸解析。
    for value_info, dims in ((onnx_model.graph.input[0], input_shape), (onnx_model.graph.output[0], output_shape)):
        shape = value_info.type.tensor_type.shape
        while shape.dim:
            shape.dim.pop()
        for dim in dims:
            shape.dim.add().dim_value = dim

    onnx_model, ok = simplify(onnx_model, overwrite_input_shapes={INPUT_NAME: input_shape})
    if not ok:
        raise RuntimeError("onnxsim 检查失败")

    # 只使用当前环境支持的 ONNX 优化 pass。
    passes = [
        p for p in [
            "extract_constant_to_initializer",
            "eliminate_deadend",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
        ]
        if p in onnxoptimizer.get_available_passes()
    ]
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx_model.graph.input[0].name = INPUT_NAME
    onnx_model.graph.output[0].name = output_name
    # shape inference 后再次写回名称和 shape，避免优化过程改动导出接口。
    for value_info, dims in ((onnx_model.graph.input[0], input_shape), (onnx_model.graph.output[0], output_shape)):
        shape = value_info.type.tensor_type.shape
        while shape.dim:
            shape.dim.pop()
        for dim in dims:
            shape.dim.add().dim_value = dim

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, ONNX_PATH)
    os.remove(raw_path)

    size = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"saved: {ONNX_PATH} ({size:.2f} MB)")
    print(f"input : {input_shape}")
    print(f"output: {output_shape}")
    print(f"passes: {len(passes)}")


if __name__ == "__main__":
    main()

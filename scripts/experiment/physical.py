import torch
from ultralytics import YOLO

from common import write_csv
from experiment_config import IMG_SIZE
from performance import build_heatmap_model, build_ssdlite_model, configure_code_cfg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROFILE_MODE = "full_model"  # "full_model" or "first_branch"


class FirstBranchHeatmap(torch.nn.Module):
    """只统计部署使用的第一个 heatmap 分支。"""

    def __init__(self, model):
        super().__init__()
        self.layers = model.backbone.layers
        self.feature_index = model.feature_indices[0]
        self.reduce = model.reduce.p2
        self.head = model.head_p2

    def forward(self, x):
        outs, feat = [], x
        for i, layer in enumerate(self.layers):
            if i > self.feature_index:
                break
            feat = outs[layer.f] if isinstance(layer.f, int) and layer.f != -1 else feat
            if not isinstance(layer.f, int):
                feat = [feat if j == -1 else outs[j] for j in layer.f]
            feat = layer(feat)
            outs.append(feat)
        return self.head(self.reduce(outs[self.feature_index]))


class DetectionInputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model([x[0]])


def run_physical(models, out_csv):
    from thop import profile

    rows = []

    for info in models:
        if info["kind"] == "official":
            model = YOLO(str(info["weight_path"])).model
            profile_mode = "official"
            effective_input_shape = f"1x3x{IMG_SIZE}x{IMG_SIZE}"

        elif info["kind"] == "ssdlite":
            model = DetectionInputWrapper(build_ssdlite_model(info))
            profile_mode = PROFILE_MODE
            effective_input_shape = "1x3x320x320"

        else:
            configure_code_cfg(info)
            model = build_heatmap_model(info)
            if PROFILE_MODE == "first_branch":
                model = FirstBranchHeatmap(model)
            elif PROFILE_MODE != "full_model":
                raise ValueError(f"未知物理性能统计模式: {PROFILE_MODE}")
            profile_mode = PROFILE_MODE
            effective_input_shape = f"1x3x{IMG_SIZE}x{IMG_SIZE}"

        model = model.to(DEVICE).eval()
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs_g = macs / 1e9
        gflops = macs_g * 2

        for module in model.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)

        for module in model.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)

        rows.append({
            "model": info["dir_name"],
            "name": info["name"],
            "params_m": params_m,
            "macs_g": macs_g,
            "gflops": gflops,
            "model_size_mb": info["weight_path"].stat().st_size / 1024 / 1024,
            "profile_mode": profile_mode,
            "input_shape": f"1x3x{IMG_SIZE}x{IMG_SIZE}",
            "effective_input_shape": effective_input_shape,
            "device": DEVICE,
        })

    write_csv(out_csv, rows)
    return rows

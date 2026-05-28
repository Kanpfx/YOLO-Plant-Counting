"""YOLOv8 特征提取与 heatmap 检测头。"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

from config import CFG


class YOLOv8Backbone(nn.Module):
    """按配置截取 YOLOv8 中间特征。"""

    def __init__(self, model, feature_indices=CFG.feature_indices):
        super().__init__()
        if len(feature_indices) != 3:
            raise ValueError("FEATURE_INDICES must contain exactly three layer indices.")
        self.feature_indices = tuple(feature_indices)
        self.layers = model.model[: max(self.feature_indices) + 1]

    def forward(self, x):
        outputs = []
        for m in self.layers:
            if isinstance(m.f, int):
                x = outputs[m.f] if m.f != -1 else x
            else:
                x = [x if j == -1 else outputs[j] for j in m.f]
            x = m(x)
            outputs.append(x)
        return tuple(outputs[i] for i in self.feature_indices)


class ChannelReduce(nn.Module):
    def __init__(self, c2, c3, c4, mid_channels=32):
        super().__init__()
        self.p2 = nn.Conv2d(c2, mid_channels, 1)
        self.p3 = nn.Conv2d(c3, mid_channels, 1)
        self.p4 = nn.Conv2d(c4, mid_channels, 1)

    def forward(self, p2, p3, p4):
        return self.p2(p2), self.p3(p3), self.p4(p4)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HeatmapHead(nn.Module):
    """单尺度 heatmap 检测头。"""

    def __init__(self, channels=32, num_classes=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1),
            ConvBNAct(channels, channels, 3, 1, 1),
        )
        self.heatmap = nn.Conv2d(channels, num_classes, 1)
        self.center = nn.Conv2d(channels, 2, 1)
        self.size = nn.Conv2d(channels, 2, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x):
        x = self.block(x)
        return torch.cat((self.heatmap(x), self.center(x), self.size(x)), dim=1)


class HeatmapYOLO(nn.Module):
    """YOLOv8 特征提取器加三尺度 heatmap head。"""

    def __init__(self, yolo_model, num_classes=1, mid_channels=32, feature_indices=CFG.feature_indices):
        super().__init__()
        self.feature_indices = tuple(feature_indices)
        self.feature_names = tuple(CFG.feature_names)
        self.backbone = YOLOv8Backbone(yolo_model, self.feature_indices)

        # 用虚拟输入推断三个尺度的通道数。
        with torch.no_grad():
            dummy = torch.zeros(1, 3, CFG.patch_size, CFG.patch_size)
            p2, p3, p4 = self.backbone(dummy)
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        self.reduce = ChannelReduce(c2, c3, c4, mid_channels)
        self.head_p2 = HeatmapHead(mid_channels, num_classes)
        self.head_p3 = HeatmapHead(mid_channels, num_classes)
        self.head_p4 = HeatmapHead(mid_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        p2, p3, p4 = self.backbone(x)
        p2, p3, p4 = self.reduce(p2, p3, p4)
        return self.head_p2(p2), self.head_p3(p3), self.head_p4(p4)

    def split_output(self, pred):
        """拆分 heatmap、中心偏移和宽高输出。"""
        nc = self.num_classes
        heatmap = pred[:, :nc, :, :]
        center = pred[:, nc:nc + 2, :, :]
        wh = pred[:, nc + 2:nc + 4, :, :]
        return heatmap, center, wh

    @torch.no_grad()
    def initial_analysis(self, x=None, input_shape=(1, 3, 1024, 1024)):
        """打印模型结构和各尺度输出形状。"""
        if x is None:
            device = next(self.parameters()).device
            x = torch.zeros(*input_shape, device=device)

        outputs, feat = [], x
        log = ["\n===== Model Structure Analysis =====", "\n========== Backbone =========="]
        for i, m in enumerate(self.backbone.layers):
            if isinstance(m.f, int):
                feat = outputs[m.f] if m.f != -1 else feat
            else:
                feat = [feat if j == -1 else outputs[j] for j in m.f]

            feat = m(feat)
            outputs.append(feat)
            log.append(f"Layer {i:02d} -> list output" if isinstance(feat, list) else f"Layer {i:02d} -> {tuple(feat.shape)}")

        # 对齐检查：截取特征、降维输出和检测头输出。
        p2, p3, p4 = (outputs[i] for i in self.feature_indices)
        log.append("\n====== Extracted Features ======")
        for name, feat in zip(self.feature_names, (p2, p3, p4)):
            log.append(f"{name}: {tuple(feat.shape)}")

        log.append("\n======== Channel Reduce ========")
        p2, p3, p4 = self.reduce(p2, p3, p4)
        for name, feat in zip(self.feature_names, (p2, p3, p4)):
            log.append(f"{name} -> {tuple(feat.shape)}")

        log.append("\n======== Heatmap Heads =========")
        out2 = self.head_p2(p2)
        out3 = self.head_p3(p3)
        out4 = self.head_p4(p4)
        h2, c2, s2 = self.split_output(out2)
        h3, c3, s3 = self.split_output(out3)
        h4, c4, s4 = self.split_output(out4)

        for name, out, hm, ctr, size in zip(
            self.feature_names,
            (out2, out3, out4),
            (h2, h3, h4),
            (c2, c3, c4),
            (s2, s3, s4),
        ):
            log.append(
                f"{name} out {tuple(out.shape)} | heatmap {tuple(hm.shape)} "
                f"center {tuple(ctr.shape)} size {tuple(size.shape)}"
            )

        log.append("\n======== Analysis Finished ========")
        print("\n".join(log))
        model_info(self, imgsz=input_shape[-1])
        return out2, out3, out4


if __name__ == "__main__":
    yolo = YOLO(CFG.yolo_yaml).model
    model = HeatmapYOLO(yolo, num_classes=CFG.num_classes, mid_channels=32)
    HeatmapYOLO.initial_analysis(model)

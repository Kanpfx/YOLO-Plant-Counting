"""CenterNet 风格检测损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetLoss(nn.Module):
    """heatmap focal loss 与中心/宽高回归 loss。"""

    def __init__(self, alpha=2, beta=4, wh_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.wh_weight = wh_weight

    def heatmap_loss(self, pred_hm, gt_hm):
        """计算类别热力图 focal loss。"""
        pred_hm = pred_hm.sigmoid().clamp(1e-4, 1 - 1e-4)
        pos_mask = (gt_hm == 1).float()
        neg_mask = (gt_hm < 1).float()
        neg_weight = torch.pow(1 - gt_hm, self.beta)

        pos_loss = -torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_mask
        neg_loss = -torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * neg_weight * neg_mask

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        num_pos = pos_mask.sum()
        if num_pos == 0:
            return neg_loss
        return (pos_loss + neg_loss) / num_pos

    def reg_l1_loss(self, pred, gt, pos_mask):
        """只在目标中心点计算 L1 回归损失。"""
        num = pos_mask.sum() + 1e-4
        mask = pos_mask.expand_as(pred).float()
        loss = F.l1_loss(pred * mask, gt * mask, reduction="sum")
        return loss / num

    def forward(self, pred, target):
        """返回总损失和分项损失。"""
        pred_hm, pred_ctr, pred_wh = pred
        gt_hm, gt_ctr, gt_wh = target

        heatmap_loss = self.heatmap_loss(pred_hm, gt_hm)
        reg_pos_mask = (gt_hm == 1).float().max(dim=1, keepdim=True).values
        center_loss = self.reg_l1_loss(pred_ctr, gt_ctr, reg_pos_mask)
        wh_loss = self.reg_l1_loss(pred_wh, gt_wh, reg_pos_mask)
        total = heatmap_loss + center_loss + self.wh_weight * wh_loss

        return total, {
            "heatmap": heatmap_loss.detach(),
            "center": center_loss.detach(),
            "wh": wh_loss.detach(),
        }

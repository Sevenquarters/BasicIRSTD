import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge


class MaskEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = torch.FloatTensor([[0, -1, 0],
                                      [0, 0, 0],
                                      [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.FloatTensor([[0, 0, 0],
                                      [-1, 0, 1],
                                      [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weight_h", kernel_h)
        self.register_buffer("weight_v", kernel_v)

    def forward(self, x):
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        edge = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return torch.clamp(edge, min=0.0, max=1.0)


class AuxShapeLoss(nn.Module):
    """Segmentation loss plus lightweight shape-aware auxiliary supervision."""
    def __init__(self, aux_weight=0.2):
        super().__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.edge_extractor = MaskEdgeExtractor()
        self.aux_weight = aux_weight

    def forward(self, preds, gt_masks):
        seg_preds, aux_shape_pred = preds
        seg_loss = self.softiou(seg_preds, gt_masks)

        edge_gt = self.edge_extractor(gt_masks.clone())
        if aux_shape_pred.shape[2:] != edge_gt.shape[2:]:
            aux_shape_pred = F.interpolate(
                aux_shape_pred,
                size=edge_gt.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

        aux_loss = self.bce(aux_shape_pred, edge_gt) + self.softiou(aux_shape_pred, edge_gt)
        return seg_loss + self.aux_weight * aux_loss

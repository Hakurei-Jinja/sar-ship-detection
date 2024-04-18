from abc import ABCMeta, abstractmethod
import math

from torch import Tensor
import torch

from ..utils import LossConfig

eps = 1e-7


class BboxIoU(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, pred_box: Tensor, target_box: Tensor) -> Tensor:
        """calculate IoU between two bounding boxes

        Args:
            pred_box: Tensor with shape [N, 4], where N is the number of bounding boxes
            target_box: Tensor with shape [N, 4], where N is the number of bounding boxes

        Returns:
            Tensor: a tensor with shape [N, 1], where N is the number of bounding boxes
        """
        pass


class IoUMetric(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def iou(box1: Tensor, box2: Tensor) -> Tensor:
        pass

    @staticmethod
    @abstractmethod
    def get_bbox(box: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        pass


class IoU(IoUMetric):
    @staticmethod
    def iou(box1: Tensor, box2: Tensor) -> Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = IoU.get_bbox(box1)
        b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = IoU.get_bbox(box2)
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)

        union = w1 * h1 + w2 * h2 - inter + eps
        return inter / union

    @staticmethod
    def get_bbox(box: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b_x1, b_y1, b_x2, b_y2 = box.chunk(4, -1)
        w, h = b_x2 - b_x1, b_y2 - b_y1 + eps
        return b_x1, b_y1, b_x2, b_y2, w, h


class InnerIoU(IoUMetric):
    ratio = 0.8

    @staticmethod
    def iou(box1: Tensor, box2: Tensor) -> Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = InnerIoU.get_inner_bbox(box1)
        b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = InnerIoU.get_inner_bbox(box2)
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)
        union = w1 * h1 + w2 * h2 - inter + eps
        return inter / union

    @staticmethod
    def get_bbox(box: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b_x1, b_y1, b_x2, b_y2 = box.chunk(4, -1)
        w, h = b_x2 - b_x1, b_y2 - b_y1 + eps
        return b_x1, b_y1, b_x2, b_y2, w, h

    @staticmethod
    def get_inner_bbox(
        box: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b_x1, b_y1, b_x2, b_y2, w, h = InnerIoU.get_bbox(box)
        b_x1 = b_x1 + w / 2 * (1 - InnerIoU.ratio)
        b_y1 = b_y1 + h / 2 * (1 - InnerIoU.ratio)
        b_x2 = b_x2 - w / 2 * (1 - InnerIoU.ratio)
        b_y2 = b_y2 - h / 2 * (1 - InnerIoU.ratio)
        w = w * InnerIoU.ratio
        h = h * InnerIoU.ratio
        return b_x1, b_y1, b_x2, b_y2, w, h


class BasicIoU(BboxIoU):
    def __init__(self, iou_metric: IoUMetric):
        self.__iou_metric = iou_metric

    def __call__(self, pred_box: Tensor, target_box: Tensor) -> Tensor:
        return self.__iou_metric.iou(pred_box, target_box)


class CIoU(BboxIoU):
    def __init__(self, iou_metric: IoUMetric):
        self.__iou_metric = iou_metric

    def __call__(self, pred_box: Tensor, target_box: Tensor) -> Tensor:
        return self.__bbox_iou(pred_box, target_box)

    def __bbox_iou(self, box1: Tensor, box2: Tensor) -> Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = self.__iou_metric.get_bbox(box1)
        b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = self.__iou_metric.get_bbox(box2)
        iou = self.__iou_metric.iou(box1, box2)
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
        ) / 4
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU


class DIoU(BboxIoU):
    def __init__(self, iou_metric: IoUMetric):
        self.__iou_metric = iou_metric

    def __call__(self, pred_box: Tensor, target_box: Tensor) -> Tensor:
        return self.__bbox_iou(pred_box, target_box)

    def __bbox_iou(self, box1: Tensor, box2: Tensor) -> Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = self.__iou_metric.get_bbox(box1)
        b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = self.__iou_metric.get_bbox(box2)
        iou = self.__iou_metric.iou(box1, box2)
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
        ) / 4
        v = 4 / (math.pi**2) * ((torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2))
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return iou - (rho2 / c2 + v * alpha)


class GIoU(BboxIoU):
    def __init__(self, iou_metric: IoUMetric):
        self.__iou_metric = iou_metric

    def __call__(self, pred_box: Tensor, target_box: Tensor) -> Tensor:
        return self.__bbox_iou(pred_box, target_box)

    def __bbox_iou(self, box1: Tensor, box2: Tensor) -> Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2, _, _ = self.__iou_metric.get_bbox(box1)
        b2_x1, b2_y1, b2_x2, b2_y2, _, _ = self.__iou_metric.get_bbox(box2)
        iou = self.__iou_metric.iou(box1, box2)
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c_area = cw * ch + eps
        return iou - (c_area - iou) / c_area


class SIoU(BboxIoU):
    def __init__(self, iou_metric: IoUMetric):
        self.__iou_metric = iou_metric

    def __call__(self, pred_box: Tensor, target_box: Tensor) -> Tensor:
        return self.__bbox_iou(pred_box, target_box)

    def __bbox_iou(self, box1: Tensor, box2: Tensor) -> Tensor:
        b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = self.__iou_metric.get_bbox(box1)
        b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = self.__iou_metric.get_bbox(box2)
        iou = self.__iou_metric.iou(box1, box2)
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
            1 - torch.exp(-1 * omiga_h), 4
        )
        return iou - 0.5 * (distance_cost + shape_cost)


class BboxIoUFactory:
    @staticmethod
    def get_iou(cfg: LossConfig) -> BboxIoU:
        loss_type = cfg.bbox_loss
        inner_ratio = cfg.inner_ratio
        if inner_ratio is not None:
            InnerIoU.ratio = inner_ratio
        if loss_type == "CIoU":
            return CIoU(IoU())
        elif loss_type == "DIoU":
            return DIoU(IoU())
        elif loss_type == "GIoU":
            return GIoU(IoU())
        elif loss_type == "SIoU":
            return SIoU(IoU())
        elif loss_type == "IoU":
            return BasicIoU(IoU())
        elif loss_type == "InnerCIoU":
            return CIoU(InnerIoU())
        elif loss_type == "InnerDIoU":
            return DIoU(InnerIoU())
        elif loss_type == "InnerGIoU":
            return GIoU(InnerIoU())
        elif loss_type == "InnerSIoU":
            return SIoU(InnerIoU())
        elif loss_type == "InnerIoU":
            return BasicIoU(InnerIoU())
        elif loss_type is None:
            return CIoU(IoU())
        else:
            raise ValueError(f"Unknown IoU type: {loss_type}")

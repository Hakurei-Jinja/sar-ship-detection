from copy import deepcopy

import torch
from torch.nn import Module
from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist
from ultralytics.utils.torch_utils import initialize_weights

from .loss import BboxIoUFactory
from .modules import Detect, OBB, Pose, Segment
from .parser import ModelParser
from .utils import LossConfig


class MyYOLO(YOLO):
    def __init__(
        self, model: str = "yolov8n.pt", task: str | None = None, verbose=False
    ):
        if task is None:
            super().__init__(model=model, verbose=verbose)
        else:
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        task_map = super().task_map
        task_map["detect"]["model"] = MyDetectionModel
        task_map["detect"]["trainer"] = MyDetectionTrainer
        return task_map


class MyDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore
        """Return a YOLO detection model."""
        model = MyDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)  # type: ignore
        if weights:
            model.load(weights)
        return model


class MyDetectionModel(DetectionModel):
    """YOLO detection model."""

    model_parser = ModelParser()

    def __init__(
        self,
        cfg: dict | str = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ):  # model, input channels, number of classes
        """Initialize the YOLO detection model with the given config and parameters."""
        Module.__init__(self)
        self.yaml = self.__get_model_cfg(cfg, ch, nc)
        self.model, self.save = self.model_parser.parse_model(
            deepcopy(self.yaml), ch=self.yaml["ch"], verbose=verbose
        )
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.__build_strides()
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def __get_model_cfg(self, cfg: dict | str, ch: int, nc: int | None) -> dict:
        """Get the model config from the given config."""
        yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        yaml["ch"] = yaml.get("ch", ch)
        if nc and yaml["nc"] != nc:
            LOGGER.info(f"Overriding model.yaml nc={yaml['nc']} with nc={nc}")
            yaml["nc"] = nc
        return yaml

    def __build_strides(self):
        """Build strides for the model."""
        m = self.model[-1]  # Detect()
        if not isinstance(m, Detect):
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
            return
        s = 256  # 2x min stride
        m.inplace = self.inplace
        forward = lambda x: (
            self.forward(x)[0]
            if isinstance(m, (Segment, Pose, OBB))
            else self.forward(x)
        )
        m.stride = torch.tensor(
            [s / x.shape[-2] for x in forward(torch.zeros(2, self.yaml["ch"], s, s))]
        )  # forward
        self.stride = m.stride
        m.bias_init()  # only run once

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return MyDetectionLoss(self, self.__get_loss_cfg(self.yaml))

    @staticmethod
    def __get_loss_cfg(cfg: dict) -> LossConfig:
        bbox_loss = cfg.get("bbox_loss", None)
        inner_ratio = cfg.get("inner_ratio", None)
        return LossConfig(bbox_loss=bbox_loss, inner_ratio=inner_ratio)


class MyDetectionLoss(v8DetectionLoss):
    def __init__(self, model: Module, loss_cfg: LossConfig):
        super().__init__(model)
        device = next(model.parameters()).device
        m = model.model[-1]
        self.bbox_loss = MyBboxLoss(
            m.reg_max - 1,
            use_dfl=self.use_dfl,
            loss_cfg=loss_cfg,
        ).to(device)


class MyBboxLoss(BboxLoss):

    def __init__(
        self,
        reg_max: int,
        use_dfl: bool = False,
        loss_cfg: LossConfig = LossConfig(),
    ):
        super().__init__(reg_max=reg_max, use_dfl=use_dfl)
        self.__bbox_iou = BboxIoUFactory.get_iou(loss_cfg)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        # IoU loss
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = self.__bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self.__get_loss_dfl(pred_dist, target_ltrb, weight, fg_mask)
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

    def __get_loss_dfl(self, pred_dist, target_ltrb, weight, fg_mask):
        pred = pred_dist[fg_mask].view(-1, self.reg_max + 1)
        target = target_ltrb[fg_mask]
        return self._df_loss(pred, target) * weight

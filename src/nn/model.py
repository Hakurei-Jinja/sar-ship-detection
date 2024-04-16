from copy import deepcopy

import torch
from ultralytics.models import YOLO
from ultralytics.nn.tasks import BaseModel, yaml_model_load
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights, scale_img
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .modules import Detect, OBB, Pose, Segment
from .parser import ModelParser


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
        task_map["detect"]["model"] = DetectionModel
        task_map["detect"]["trainer"] = MyDetectionTrainer
        return task_map


class MyDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)  # type: ignore
        if weights:
            model.load(weights)
        return model


class DetectionModel(BaseModel):
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
        super().__init__()
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
            [s / x.shape[-2] for x in forward(torch.zeros(1, self.yaml["ch"], s, s))]
        )  # forward
        self.stride = m.stride
        m.bias_init()  # only run once

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)

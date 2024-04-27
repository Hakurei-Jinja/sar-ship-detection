from dataclasses import asdict, dataclass
import os

from src.nn.model import MyYOLO


@dataclass
class TrainConfig:
    """
    default values are from ultralytics/cfg/default.yaml
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml for more details
    """

    epochs: int = 100
    time: float | None = None
    patience: int | None = 100
    batch: int = 16
    imgsz: int = 640
    save: bool = True
    save_period: int = -1
    cache: bool = False
    device: int | list[int] | str | None = None
    workers: int = 8
    project: str | None = None
    name: str | None = None
    exist_ok: bool = False
    pretrained: bool = True
    optimizer: str = "auto"
    verbose: bool = True
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    resume: bool = False
    amp: bool = True
    fraction: float = 1.0
    profile: bool = False
    freeze: int | list[int] | None = None
    multi_scale: bool = False
    # segmentation
    overlap_mask: bool = True
    mask_ratio: int = 4
    # classification
    dropout: float = 0.0

    dict = asdict


@dataclass
class HyperParameters:
    """
    default values are from ultralytics/cfg/default.yaml
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml for more details
    """

    # optimizer
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    # loss weights
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    pose: float = 12.0
    kobj: float = 1.0
    # data enhancement
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    auto_augment: str = "randaugment"
    erasing: float = 0.4
    crop_fraction: float = 1.0
    # other
    label_smoothing: float = 0.0
    nbs: int = 64

    dict = asdict


class Trainer:
    __train_cfg: TrainConfig
    __hyper_params: HyperParameters

    def __init__(
        self, data: str, train_cfg: TrainConfig, hyper_params: HyperParameters
    ):
        self.__data = data
        self.__train_cfg = train_cfg
        self.__hyper_params = hyper_params

    def train(self, paths: list[str] | str):
        paths = self.__to_list(paths)
        for path in paths:
            self.__check_model(path)
        for path in paths:
            self.__train(path)

    @staticmethod
    def __to_list(path: list[str] | str) -> list[str]:
        if isinstance(path, str):
            return [path]
        return path

    @staticmethod
    def __check_model(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} not found")
        try:
            MyYOLO(path, verbose=False)
        except Exception as e:
            raise e

    def __train(self, path: str):
        model = MyYOLO(path, verbose=True)
        model.train(
            data=self.__data, **self.__train_cfg.dict(), **self.__hyper_params.dict()
        )


ssdd_train_cfg = TrainConfig(
    single_cls=True,
    epochs=2000,
    batch=64,
)
ssdd_hyper_params = HyperParameters(
    hsv_h=0,
    hsv_s=0,
    translate=0,
    erasing=0.2,
)

if __name__ == "__main__":
    ssdd_trainer = Trainer(
        "./datasets/SSDD/cfg/detect/ssdd_all.yaml", ssdd_train_cfg, ssdd_hyper_params
    )
    ssdd_obb_trainer = Trainer(
        "./datasets/SSDD/cfg/obb/ssdd_all_obb.yaml", ssdd_train_cfg, ssdd_hyper_params
    )
    ssdd_seg_trainer = Trainer(
        "./datasets/SSDD/cfg/seg/ssdd_all_seg.yaml", ssdd_train_cfg, ssdd_hyper_params
    )

    ssdd_trainer.train(
        [
            # "./models/cfg/detect/v8n.yaml",
            # "./models/cfg/detect/v8n-sa.yaml",
            "./models/cfg/detect/v8n-dc.yaml",
            "./models/cfg/detect/v8n-sa-dc.yaml",
            # "./models/cfg/detect/v8s.yaml",
        ]
    )
    # ssdd_obb_trainer.train(
    #     [
    #         "./models/cfg/obb/v8n.yaml",
    #         "./models/cfg/obb/v8s.yaml",
    #     ]
    # )
    # ssdd_seg_trainer.train(
    #     [
    #         "./models/cfg/seg/v8n.yaml",
    #         "./models/cfg/seg/v8s.yaml",
    #     ]
    # )

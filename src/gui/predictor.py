from dataclasses import asdict, dataclass

from ultralytics.engine.model import Model

from .utils.np_image import NPImage


@dataclass
class PredictorConfig:
    conf: float
    iou: float
    augment: bool
    show_labels: bool
    show_conf: bool
    save: bool

    __getitem__ = lambda self, key: getattr(self, key)
    __setitem__ = lambda self, key, value: setattr(self, key, value)
    dict = asdict


class NPImageProcessError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


class Predictor:
    __model: Model

    def __init__(self, model: Model):
        self.__model = model

    def predict(self, image: NPImage, cfg: PredictorConfig) -> NPImage:
        self.__check_image(image)
        results = self.__model.predict(
            image.get_image(),
            conf=cfg.conf,
            iou=cfg.iou,
            augment=cfg.augment,
            save=cfg.save,
        )
        np_image = self.__get_np_image_from_results(results, cfg)
        np_image.BGR2RGB()
        return np_image

    def __check_image(self, image: NPImage):
        _, _, channel = image.get_image().shape
        if channel != 3:
            raise NPImageProcessError("Image type must be RGB")

    def __get_np_image_from_results(self, results: list, cfg: PredictorConfig):
        result = results[0]  # Get the first image from the results
        np_arr = result.plot(conf=cfg.show_conf, labels=cfg.show_labels).astype("uint8")
        return NPImage(np_arr)

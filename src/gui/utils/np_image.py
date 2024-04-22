from abc import ABCMeta, abstractmethod

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap
import numpy as np
from ultralytics.engine.model import Model


class NPImage:
    __image: np.ndarray

    def __init__(self, image: np.ndarray | Image.Image):
        if isinstance(image, Image.Image):
            self.__image = np.array(image)
        elif isinstance(image, np.ndarray):
            self.__image = image
        else:
            raise ValueError("Image must be of type np.ndarray or PIL.Image.Image")

    def get_image(self):
        return self.__image

    def get_PIL_image(self):
        return Image.fromarray(self.__image).convert("RGB")

    def get_Qt_image(self):
        return ImageQt(self.get_PIL_image())

    def get_Qt_pixmap(self):
        return QPixmap.fromImage(self.get_Qt_image())

    def BGR2RGB(self):
        self.__image = self.__image[..., ::-1]

    def RGB2BGR(self):
        self.BGR2RGB()


class NPImageProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, image: NPImage) -> NPImage:
        pass


class NPImageProcessError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


class NPImagePredictor(NPImageProcessor):
    __model: Model

    def __init__(self, model: Model):
        self.__model = model

    def process(self, image: NPImage) -> NPImage:
        self.__check_image(image)
        results = self.__model.predict(image.get_image(), show_labels=False)
        np_image = self.__get_np_image_from_results(results)
        np_image.BGR2RGB()
        return np_image

    def __check_image(self, image: NPImage):
        _, _, channel = image.get_image().shape
        if channel != 3:
            raise NPImageProcessError("Image type must be RGB")

    def __get_np_image_from_results(self, results: list):
        result = results[0]  # Get the first image from the results
        np_arr = result.plot().astype("uint8")
        return NPImage(np_arr)

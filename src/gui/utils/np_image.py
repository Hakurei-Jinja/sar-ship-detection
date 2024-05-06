from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap
import numpy as np


class NPImage:
    __image: np.ndarray

    def __init__(self, image: np.ndarray | Image.Image):
        if isinstance(image, Image.Image):
            self.__image = np.array(image)
        elif isinstance(image, np.ndarray):
            self.__image = image
        else:
            raise ValueError("Image must be of type np.ndarray or PIL.Image.Image")

    def get_shape(self):
        height, width, channel = self.__image.shape
        return {"height": height, "width": width, "channel": channel}

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

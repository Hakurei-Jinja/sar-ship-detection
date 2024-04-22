from abc import ABCMeta, abstractmethod
import os
import shutil


class ImageConvertor(metaclass=ABCMeta):
    @abstractmethod
    def convert_images(
        self, source_path: str, destination_path: str, img_name_list: list[str]
    ):
        pass


class ImageCopyConvertor(ImageConvertor):
    def convert_images(
        self, source_path: str, destination_path: str, img_name_list: list[str]
    ):
        os.makedirs(destination_path, exist_ok=True)
        images = os.listdir(source_path)
        for image in images:
            if image not in img_name_list:
                continue
            image_path = os.path.join(source_path, image)
            shutil.copy(image_path, destination_path)


class ImageConvertorFactory:
    def get_convertor(self, label_type: str) -> ImageConvertor:
        if label_type == "voc":
            return ImageCopyConvertor()
        elif label_type == "voc_obb":
            return ImageCopyConvertor()
        elif label_type == "voc_seg":
            return ImageCopyConvertor()
        elif label_type == "coco":
            return ImageCopyConvertor()
        raise ValueError("Invalid label type")

import os

import yaml

from .image_convertor import ImageConvertor, ImageConvertorFactory
from .label_convertor import LabelConvertor, LabelConvertorFactory
from .image_lister import ImageLister, ImageListerFactory


class DatasetConvertor:
    _config: list
    __name_id_dict: dict[str, int] | None
    __label_convertor_factory: LabelConvertorFactory
    __image_convertor_factory: ImageConvertorFactory
    __image_lister_factory: ImageListerFactory

    def __init__(self, config_path: str = ""):
        self._config = []
        self.__name_id_dict = None
        if config_path:
            self.load_config(config_path)
        self.__label_convertor_factory = LabelConvertorFactory()
        self.__image_convertor_factory = ImageConvertorFactory()
        self.__image_lister_factory = ImageListerFactory()

    def load_config(self, config_path: str, class_file_path: str | None = None):
        with open(config_path, "r") as file:
            self._config = self.__to_list(yaml.safe_load(file))
        if not class_file_path:
            return
        with open(class_file_path, "r") as file:
            self.__name_id_dict = self.__to_class_id_dict(yaml.safe_load(file))

    def __to_list(self, data: dict | list) -> list:
        if isinstance(data, dict):
            return [data]
        return data

    def __to_class_id_dict(self, yaml_data: dict) -> dict:
        id_name_dict = yaml_data["names"]
        return {name: id for id, name in id_name_dict.items()}

    def convert(self):
        self.__check_config()
        for each in self._config:
            img_source_path, img_destination_path = self.__get_images_path(each)
            label_source_path, label_destination_path = self.__get_labels_path(each)

            image_convertor = self.__get_image_convertor(each)
            img_list = self.__get_image_lister(each)
            label_convertor = self.__get_label_convertor(each)

            image_convertor.convert_images(
                img_source_path,
                img_destination_path,
                img_list.get_img_name_list(label_source_path),
            )
            label_convertor.convert_labels(
                label_source_path, label_destination_path, self.__name_id_dict
            )

    def __check_config(self):
        if not self._config:
            raise NameError("Config is empty")

    def __get_images_path(self, config: dict) -> tuple[str, str]:
        source_path = config["img"]
        destination_path = os.path.join(config["out"], "images", config["data_type"])
        return source_path, destination_path

    def __get_labels_path(self, config: dict) -> tuple[str, str]:
        source_path = config["label"]
        destination_path = os.path.join(config["out"], "labels", config["data_type"])
        return source_path, destination_path

    def __get_label_convertor(self, config: dict) -> LabelConvertor:
        label_type = config["label_type"]
        return self.__label_convertor_factory.get_convertor(label_type)

    def __get_image_convertor(self, config: dict) -> ImageConvertor:
        label_type = config["label_type"]
        return self.__image_convertor_factory.get_convertor(label_type)

    def __get_image_lister(self, config: dict) -> ImageLister:
        label_type = config["label_type"]
        return self.__image_lister_factory.get_lister(label_type)

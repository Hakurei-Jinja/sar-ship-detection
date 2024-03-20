import os
import shutil

import yaml

from .label_convertor import LabelConvertor, LabelConvertorFactory


class DatasetConvertor:
    _config: list
    __label_convertor_factory: LabelConvertorFactory

    def __init__(self, config_path: str = ""):
        self._config = []
        if config_path:
            self.load_config(config_path)
        self.__label_convertor_factory = LabelConvertorFactory()

    def load_config(self, config_path: str):
        with open(config_path, "r") as file:
            self._config = self.__to_list(yaml.safe_load(file))

    def __to_list(self, data: dict | list) -> list:
        if isinstance(data, dict):
            return [data]
        return data

    def convert(self):
        self.__check_config()
        for each in self._config:
            source_dir, destination_dir = self.__get_images_dir(each)
            self.__copy_images(source_dir, destination_dir)

            source_dir, destination_dir = self.__get_labels_dir(each)
            label_convertor = self.__get_label_convertor(each)
            self.__convert_labels(source_dir, destination_dir, label_convertor)

    def __check_config(self):
        if not self._config:
            raise NameError("Config is empty")

    def __get_images_dir(self, config: dict) -> tuple[str, str]:
        source_dir = config["img_dir"]
        destination_dir = os.path.join(config["out_dir"], "images", config["data_type"])
        return source_dir, destination_dir

    def __copy_images(self, source_dir: str, destination_dir: str):
        os.makedirs(destination_dir, exist_ok=True)
        images = os.listdir(source_dir)
        for image in images:
            image_path = os.path.join(source_dir, image)
            shutil.copy(image_path, destination_dir)

    def __get_labels_dir(self, config: dict) -> tuple[str, str]:
        source_dir = config["label_dir"]
        destination_dir = os.path.join(config["out_dir"], "labels", config["data_type"])
        return source_dir, destination_dir

    def __get_label_convertor(self, config: dict) -> LabelConvertor:
        label_type = config["label_type"]
        return self.__label_convertor_factory.get_convertor(label_type)

    def __convert_labels(
        self, source_dir: str, destination_dir: str, label_convertor: LabelConvertor
    ):
        os.makedirs(destination_dir, exist_ok=True)
        files = os.listdir(source_dir)
        for file in files:
            source_file_path = os.path.join(source_dir, file)
            destination_file_path = os.path.join(
                destination_dir, self.__get_file_name(file) + ".txt"
            )
            with open(source_file_path, "r") as f:
                file_content = label_convertor.convert(f)
            with open(destination_file_path, "w") as f:
                f.write(file_content)

    def __get_file_name(self, file: str) -> str:
        return os.path.splitext(file)[0]

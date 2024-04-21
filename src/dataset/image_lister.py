from abc import ABCMeta, abstractmethod
from io import TextIOWrapper
import json
import os

from .utils.xml_parser import XMLParser


class ImageLister(metaclass=ABCMeta):
    @abstractmethod
    def get_img_name_list(self, source_path: str) -> list[str]:
        pass


class VOCImageLister(ImageLister):
    def __init__(self):
        self.__xml_parser = XMLParser()

    def get_img_name_list(self, source_path: str) -> list[str]:
        img_name_list = []
        files = os.listdir(source_path)
        for file in files:
            source_file_path = os.path.join(source_path, file)
            with open(source_file_path, "r") as f:
                img_name_list.append(self.__get_img_name(f))
        return img_name_list

    def __get_img_name(self, file: TextIOWrapper) -> str:
        xml_root = self.__xml_parser.get_xml_root(file)
        return self.__xml_parser.get_xml_text(xml_root, "filename")


class COCOImageLister(ImageLister):

    def get_img_name_list(self, source_path: str) -> list[str]:
        cfg = self.__get_cfg(source_path)
        return [x["file_name"] for x in cfg["images"]]

    def __get_cfg(self, source_path: str) -> dict:
        with open(source_path, "r") as f:
            return json.load(f)


class ImageListerFactory:
    @staticmethod
    def get_lister(dataset_type: str) -> ImageLister:
        if dataset_type == "voc":
            return VOCImageLister()
        if dataset_type == "voc_obb":
            return VOCImageLister()
        elif dataset_type == "coco":
            return COCOImageLister()
        raise ValueError("Invalid label type")

from abc import ABCMeta, abstractmethod
from io import TextIOWrapper
import xml.etree.ElementTree as ET

from .utils.xml_parser import XMLParser


class LabelConvertor(metaclass=ABCMeta):
    @abstractmethod
    def convert(self, file: TextIOWrapper) -> str:
        pass


class VOCLabelConvertor(LabelConvertor):
    __xml_parser: XMLParser

    def __init__(self):
        self.__xml_parser = XMLParser()

    def convert(self, file: TextIOWrapper) -> str:
        xml_root = self.__xml_parser.get_xml_root(file)

        converted_file = ""
        size = self.__get_size(xml_root)
        for obj in self.__xml_parser.get_xml_iter(xml_root, "object"):
            if not self.__is_obj_valid(obj):
                continue
            box = self.__get_box(obj)
            yolo_box = self.__convert_to_yolo_box(size, box)
            converted_file = converted_file + self.__generate_yolo_label(yolo_box)
        return converted_file

    def __get_size(self, xml_root: ET.Element) -> dict[str, int]:
        xml_size = self.__xml_parser.get_xml_element(xml_root, "size")
        return {
            x: int(self.__xml_parser.get_xml_text(xml_size, x))
            for x in ("width", "height")
        }

    def __is_obj_valid(self, obj: ET.Element) -> bool:
        cls = self.__xml_parser.get_xml_text(obj, "name")
        difficult = int(self.__xml_parser.get_xml_text(obj, "difficult"))
        return (
            cls == "ship" and difficult != 1
        )  # difficult equals 1 means the object is difficult to recognize

    def __get_box(self, obj: ET.Element) -> dict[str, float]:
        xml_box = self.__xml_parser.get_xml_element(obj, "bndbox")
        return {
            x: float(self.__xml_parser.get_xml_text(xml_box, x))
            for x in ("xmin", "xmax", "ymin", "ymax")
        }

    def __convert_to_yolo_box(
        self, size: dict[str, int], box: dict[str, float]
    ) -> tuple[float, float, float, float]:
        dw, dh = 1.0 / size["width"], 1.0 / size["height"]
        x, y, w, h = (
            (box["xmin"] + box["xmax"]) / 2.0 - 1,
            (box["ymin"] + box["ymax"]) / 2.0 - 1,
            box["xmax"] - box["xmin"],
            box["ymax"] - box["ymin"],
        )
        return x * dw, y * dh, w * dw, h * dh

    def __generate_yolo_label(self, yolo_box: tuple[float, float, float, float]) -> str:
        return (
            " ".join(str(a) for a in (0, *yolo_box)) + "\n"
        )  # 0 is the class id, ship


class LabelConvertorFactory:
    def get_convertor(self, label_type: str) -> LabelConvertor:
        if label_type == "voc":
            return VOCLabelConvertor()
        raise ValueError("Invalid label type")

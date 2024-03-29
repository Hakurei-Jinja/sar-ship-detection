from abc import ABCMeta, abstractmethod
from io import TextIOWrapper
import json
import os
import xml.etree.ElementTree as ET

from .utils.xml_parser import XMLParser


class LabelConvertor(metaclass=ABCMeta):
    @abstractmethod
    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):
        pass

    @abstractmethod
    def get_img_name_list(self, source_path: str) -> list[str]:
        pass


class VOCLabelConvertor(LabelConvertor):
    __xml_parser: XMLParser
    __class_id_dict: dict[str, int]

    def __init__(self):
        self.__xml_parser = XMLParser()
        self.__class_id_dict = {}

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

    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):
        self.__check_load_class_id_dict(cls_id_dict)
        self.__convert_from_path(source_path, destination_path)

    def __check_load_class_id_dict(self, name_id_dict: dict[str, int] | None):
        if not name_id_dict:
            raise NameError("VOC label convertor needs class id file")
        self.__class_id_dict = name_id_dict

    def __convert_from_path(self, source_path: str, destination_path: str):
        os.makedirs(destination_path, exist_ok=True)
        files = os.listdir(source_path)
        for file in files:
            source_file_path = os.path.join(source_path, file)
            destination_file_path = os.path.join(
                destination_path, self.__get_file_name(file) + ".txt"
            )
            with open(source_file_path, "r") as f:
                file_content = self.__convert_from_file(f)
            with open(destination_file_path, "w") as f:
                f.write(file_content)

    def __get_file_name(self, file: str) -> str:
        return os.path.splitext(file)[0]

    def __convert_from_file(self, file: TextIOWrapper) -> str:
        xml_root = self.__xml_parser.get_xml_root(file)

        converted_file = ""
        size = self.__get_size(xml_root)
        for obj in self.__xml_parser.get_xml_iter(xml_root, "object"):
            if not self.__is_obj_valid(obj):
                continue
            box = self.__get_box(obj)
            yolo_box = self.__convert_to_yolo_box(size, box)
            class_name = self.__xml_parser.get_xml_text(obj, "name")
            class_id = self.__class_id_dict[class_name]
            converted_file = converted_file + self.__generate_yolo_label(
                class_id, yolo_box
            )
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

    def __generate_yolo_label(
        self, class_id: int, yolo_box: tuple[float, float, float, float]
    ) -> str:
        return " ".join(str(a) for a in (class_id, *yolo_box)) + "\n"


class COCOLabelConvertor(LabelConvertor):
    __file_map: dict[str, list[str]]
    __img_map: dict[int, dict]
    __id_remap: dict[int, int]

    def __init__(self):
        self.__file_map = {}
        self.__img_map = {}
        self.__id_remap = {}

    def get_img_name_list(self, source_path: str) -> list[str]:
        cfg = self.__get_cfg(source_path)
        return [x["file_name"] for x in cfg["images"]]

    def __get_cfg(self, source_path: str) -> dict:
        with open(source_path, "r") as f:
            return json.load(f)

    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):
        cfg = self.__get_cfg(source_path)
        if cls_id_dict:
            self.__remap_id(cfg, cls_id_dict)
        self.__img_map = self.__get_img_map(cfg)
        for annotation in cfg["annotations"]:
            self.__convert_single_label(annotation)
        self.__write_file_map(destination_path)

    def __remap_id(self, cfg: dict, cls_id_dict: dict[str, int]):
        categories = cfg["categories"]
        self.__id_remap = {x["id"]: cls_id_dict[x["name"]] for x in categories}

    def __get_img_map(self, cfg: dict) -> dict[int, dict]:
        img_map = {}
        for img in cfg["images"]:
            img_map[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
        return img_map

    def __convert_single_label(self, annotation: dict):
        img_dict = self.__img_map[annotation["image_id"]]
        label_name = self.__get_file_name(img_dict["file_name"]) + ".txt"

        size = self.__get_size(img_dict)
        box = self.__get_box(annotation)
        x, y, w, h = self.__convert_to_yolo_box(size, box)
        cls = self.__get_class_id(annotation)

        self.__set_file_map(label_name, f"{cls} {x} {y} {w} {h}\n")

    def __write_file_map(self, destination_path: str):
        os.makedirs(destination_path, exist_ok=True)
        for label_name in self.__file_map:
            with open(os.path.join(destination_path, label_name), "w") as f:
                f.writelines(self.__file_map[label_name])

    def __get_file_name(self, file: str) -> str:
        return os.path.splitext(file)[0]

    def __get_size(self, img_dict: dict) -> dict[str, int]:
        return {"width": img_dict["width"], "height": img_dict["height"]}

    def __get_box(self, annotation: dict) -> dict[str, float]:
        return {
            x: annotation["bbox"][i]
            for i, x in enumerate(["x_center", "y_center", "width", "height"])
        }

    def __convert_to_yolo_box(
        self, size: dict[str, int], box: dict[str, float]
    ) -> tuple[float, float, float, float]:
        dw, dh = 1.0 / size["width"], 1.0 / size["height"]
        x, y, w, h = (
            box["x_center"] + box["width"] / 2,
            box["y_center"] + box["height"] / 2,
            box["width"],
            box["height"],
        )
        return x * dw, y * dh, w * dw, h * dh

    def __get_class_id(self, annotation: dict) -> int:
        if self.__id_remap:
            return self.__id_remap[annotation["category_id"]]
        return annotation["category_id"]

    def __set_file_map(self, name: str, content: str):
        if name not in self.__file_map:
            self.__file_map[name] = []
        self.__file_map[name].append(content)


class LabelConvertorFactory:
    def get_convertor(self, label_type: str) -> LabelConvertor:
        if label_type == "voc":
            return VOCLabelConvertor()
        elif label_type == "coco":
            return COCOLabelConvertor()
        raise ValueError("Invalid label type")

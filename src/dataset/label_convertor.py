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


class VOCLabelConvertor(LabelConvertor):
    __xml_parser: XMLParser
    __class_id_dict: dict[str, int] | None

    def __init__(self):
        self.__xml_parser = XMLParser()
        self.__class_id_dict = None

    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):
        self.__class_id_dict = cls_id_dict
        self.__check_class_id_dict()
        self.__convert_from_path(source_path, destination_path)

    def __check_class_id_dict(self):
        if not self.__class_id_dict:
            raise NameError("VOC label convertor needs class id file")

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
            class_id = self.__class_id_dict[class_name]  # type: ignore
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


class VOCOBBLabelConvertor(LabelConvertor):
    __xml_parser: XMLParser
    __class_id_dict: dict[str, int] | None

    def __init__(self):
        self.__xml_parser = XMLParser()
        self.__class_id_dict = None

    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):
        self.__class_id_dict = cls_id_dict
        self.__check_class_id_dict()
        self.__convert_from_path(source_path, destination_path)

    def __check_class_id_dict(self):
        if not self.__class_id_dict:
            raise NameError("VOC label convertor needs class id file")

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
            class_id = self.__class_id_dict[class_name]  # type: ignore
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
        xml_box = self.__xml_parser.get_xml_element(obj, "rotated_bndbox")
        return {
            x: float(self.__xml_parser.get_xml_text(xml_box, x))
            for x in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
        }

    def __convert_to_yolo_box(
        self, size: dict[str, int], box: dict[str, float]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        dw, dh = 1.0 / size["width"], 1.0 / size["height"]
        # fmt: off
        return (
            box["x1"] * dw, box["y1"] * dh,
            box["x2"] * dw, box["y2"] * dh,
            box["x3"] * dw, box["y3"] * dh,
            box["x4"] * dw, box["y4"] * dh,
        )
        # fmt: on

    def __generate_yolo_label(
        self,
        class_id: int,
        yolo_box: tuple[float, float, float, float, float, float, float, float],
    ) -> str:
        return " ".join(str(a) for a in (class_id, *yolo_box)) + "\n"


class VOCSEGLabelConvertor(LabelConvertor):
    __xml_parser: XMLParser
    __class_id_dict: dict[str, int] | None

    def __init__(self):
        self.__xml_parser = XMLParser()
        self.__class_id_dict = None

    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):
        self.__class_id_dict = cls_id_dict
        self.__check_class_id_dict()
        self.__convert_from_path(source_path, destination_path)

    def __check_class_id_dict(self):
        if not self.__class_id_dict:
            raise NameError("VOC label convertor needs class id file")

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
            class_id = self.__class_id_dict[class_name]  # type: ignore
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
        xml_box = self.__xml_parser.get_xml_element(obj, "segm")
        points = self.__xml_parser.get_xml_iter(xml_box, "point")
        box = {}
        for i, point in enumerate(points):
            x, y = self.__xml_parser.get_text(point).replace(" ", "").split(",")
            box[f"x{i}"], box[f"y{i}"] = float(x), float(y)
        return box

    def __convert_to_yolo_box(
        self, size: dict[str, int], box: dict[str, float]
    ) -> tuple:
        dw, dh = 1.0 / size["width"], 1.0 / size["height"]
        yolo_box = []
        for i in range(len(box) // 2):
            yolo_box.extend([box[f"x{i}"] * dw, box[f"y{i}"] * dh])
        return tuple(yolo_box)

    def __generate_yolo_label(self, class_id: int, yolo_box: tuple) -> str:
        return " ".join(str(a) for a in (class_id, *yolo_box)) + "\n"


class COCOLabelConvertor(LabelConvertor):
    __img_map: dict[int, dict]
    __annotations: list[dict]
    __id_remap: dict[int, int] | None

    def __init__(self):
        self.__img_map = {}
        self.__annotations = []
        self.__id_remap = None

    def convert_labels(
        self,
        source_path: str,
        destination_path: str,
        cls_id_dict: dict[str, int] | None = None,
    ):

        cfg = self.__get_cfg(source_path)
        self.__img_map = self.__get_img_map(cfg)
        self.__annotations = cfg["annotations"]
        if cls_id_dict:
            self.__id_remap = self.__get_id_remap(cfg, cls_id_dict)
        self.__convert_labels(destination_path)

    def __get_cfg(self, source_path: str) -> dict:
        with open(source_path, "r") as f:
            return json.load(f)

    def __get_img_map(self, cfg: dict) -> dict[int, dict]:
        img_map = {}
        for img in cfg["images"]:
            img_map[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
        return img_map

    def __get_id_remap(self, cfg: dict, cls_id_dict: dict[str, int]):
        categories = cfg["categories"]
        return {x["id"]: cls_id_dict[x["name"]] for x in categories}

    def __convert_labels(self, destination_path: str):
        files = {}
        for annotation in self.__annotations:
            name, content = self.__convert_single_label(annotation)
            if name not in files:
                files[name] = []
            files[name].append(content)
        self.__write_file_map(files, destination_path)

    def __convert_single_label(
        self,
        annotation: dict,
    ) -> tuple[str, str]:
        img_dict = self.__img_map[annotation["image_id"]]
        label_name = self.__get_file_name(img_dict["file_name"]) + ".txt"

        size = self.__get_size(img_dict)
        box = self.__get_box(annotation)
        cls = self.__get_class_id(annotation)
        x, y, w, h = self.__convert_to_yolo_box(size, box)
        content = f"{cls} {x} {y} {w} {h}\n"
        return label_name, content

    def __get_file_name(self, file: str) -> str:
        return os.path.splitext(file)[0]

    def __get_size(self, img_dict: dict) -> dict[str, int]:
        return {"width": img_dict["width"], "height": img_dict["height"]}

    def __get_box(self, annotation: dict) -> dict[str, float]:
        return {
            x: annotation["bbox"][i]
            for i, x in enumerate(["x_center", "y_center", "width", "height"])
        }

    def __get_class_id(self, annotation: dict) -> int:
        if self.__id_remap:
            return self.__id_remap[annotation["category_id"]]
        return annotation["category_id"]

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

    def __write_file_map(self, files: dict[str, list[str]], destination_path: str):
        os.makedirs(destination_path, exist_ok=True)
        for name in files.keys():
            with open(os.path.join(destination_path, name), "w") as f:
                f.writelines(files[name])


class LabelConvertorFactory:
    def get_convertor(self, label_type: str) -> LabelConvertor:
        if label_type == "voc":
            return VOCLabelConvertor()
        elif label_type == "voc_obb":
            return VOCOBBLabelConvertor()
        elif label_type == "voc_seg":
            return VOCSEGLabelConvertor()
        elif label_type == "coco":
            return COCOLabelConvertor()
        raise ValueError("Invalid label type")

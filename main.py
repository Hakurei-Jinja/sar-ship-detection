import sys

from PySide6.QtWidgets import QApplication
from ultralytics import YOLO

from src.dataset.dataset_convertor import DatasetConvertor
from src.gui.utils.np_image import NPImagePredictor
from src.gui.window import Window


def convert_SSDD():
    dataset_convertor = DatasetConvertor()
    dataset_convertor.load_config("./datasets/SSDD/cfg/convertor.yaml")
    dataset_convertor.convert()


if __name__ == "__main__":
    model = YOLO("./models/trained/yolov8s/weights/best.pt")
    image_processor = NPImagePredictor(model)
    app = QApplication(sys.argv)
    form = Window(image_processor)
    form.show()
    sys.exit(app.exec())

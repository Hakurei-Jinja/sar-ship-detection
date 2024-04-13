import sys

from PySide6.QtWidgets import QApplication

from src.dataset import DatasetConvertor
from src.gui.utils import NPImagePredictor
from src.gui.window import Window
from src.nn.model import MyYOLO


def convert_SSDD():
    dataset_convertor = DatasetConvertor()
    dataset_convertor.load_config(
        "./datasets/SSDD/cfg/convertor.yaml", "./datasets/class.yaml"
    )
    dataset_convertor.convert()


def convert_HRSID():
    dataset_convertor = DatasetConvertor()
    dataset_convertor.load_config(
        "./datasets/HRSID_png/cfg/convertor.yaml", "./datasets/class.yaml"
    )
    dataset_convertor.convert()


if __name__ == "__main__":
    # model = MyYOLO("./models/cfg/yolov9c.yaml", verbose=True)
    model = MyYOLO("yolov8.yaml", verbose=True)

    # result = model.train(data="./datasets/HRSID_png/cfg/hrsid_all.yaml", epochs=2000)

    # metrics = model.val(data="./datasets/SSDD/cfg/ssdd_all.yaml")
    # metrics = model.val(data="./datasets/SSDD/cfg/ssdd_inshore.yaml")
    # metrics = model.val(data="./datasets/SSDD/cfg/ssdd_offshore.yaml")

    result = model.train(data="./datasets/SSDD/cfg/ssdd_all.yaml")

    # app = QApplication(sys.argv)
    # form = Window(NPImagePredictor(model))
    # form.show()
    # sys.exit(app.exec())

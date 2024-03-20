from src.dataset.dataset_convertor import DatasetConvertor
from ultralytics import YOLO


def convert_SSDD():
    dataset_convertor = DatasetConvertor()
    dataset_convertor.load_config("./datasets/SSDD/cfg/convertor.yaml")
    dataset_convertor.convert()


if __name__ == "__main__":
    model = YOLO("./models/yolov9c.pt")
    model.info()
    results = model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml", epochs=200, device=0, plots=True
    )

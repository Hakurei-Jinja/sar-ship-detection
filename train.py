from src.nn.model import MyYOLO


if __name__ == "__main__":
    model = MyYOLO("./models/cfg/yolov8.yaml", verbose=True)
    result = model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml", epochs=2000
    )  # will stop training if no improvement in 100 epochs

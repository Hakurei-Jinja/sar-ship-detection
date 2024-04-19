from src.nn.model import MyYOLO


if __name__ == "__main__":
    model = MyYOLO("./models/cfg/yolov8.yaml", verbose=True)
    result = model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml",
        epochs=2000,
        batch=-1,
        optimizer="AdamW",
    )  # will stop training if no improvement in 100 epochs
    model = MyYOLO("./models/cfg/yolov8_sa.yaml", verbose=True)
    result = model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml",
        epochs=2000,
        batch=-1,
        optimizer="AdamW",
    )  # will stop training if no improvement in 100 epochs
    model = MyYOLO("./models/cfg/yolov8_sppfcspc.yaml", verbose=True)
    result = model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml",
        epochs=2000,
        batch=-1,
        optimizer="AdamW",
    )  # will stop training if no improvement in 100 epochs
    model = MyYOLO("./models/cfg/yolov8_sa_sppfcspc.yaml", verbose=True)
    result = model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml",
        epochs=2000,
        batch=-1,
        optimizer="AdamW",
    )  # will stop training if no improvement in 100 epochs

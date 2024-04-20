from src.nn.model import MyYOLO
import os


def train_ssdd(path: str):
    model = MyYOLO(path, verbose=True)
    model.train(
        # data
        data="./datasets/SSDD/cfg/ssdd_all.yaml",
        single_cls=True,
        # epochs
        epochs=1000,
        patience=None,
        batch=100,
        # loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # optimizer
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # prevent overfitting
        weight_decay=0.0005,
        # data enhancement
        ## disable some data enhancement
        hsv_h=0,
        hsv_s=0,
        translate=0,
        ## enabled data enhancement
        mosaic=1,
        close_mosaic=10,  # close mosaic in last 10 epochs
        hsv_v=0.4,
        degrees=45,
        scale=0.5,
        erasing=0.2,
        fliplr=0.5,
        # others
        cache=True,
        val=True,
        plots=True,
    )


def train_models(path: list[str]):
    for p in path:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path {p} not found")
        MyYOLO(p, verbose=False)
    for p in path:
        train_ssdd(p)


if __name__ == "__main__":
    train_models(
        [
            "./models/cfg/yolov8.yaml",
            "./models/cfg/yolov8_sa_fix.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou01.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou02.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou03.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou04.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou05.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou06.yaml",
        ]
    )

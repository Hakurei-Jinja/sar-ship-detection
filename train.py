from src.nn.model import MyYOLO
import os


def train_ssdd(path: str):
    model = MyYOLO(path, verbose=True)
    model.train(
        data="./datasets/SSDD/cfg/ssdd_all.yaml", epochs=2000, batch=100
    )  # will stop training if no improvement in 100 epochs


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
            "./models/cfg/yolov8_sa.yaml",
            "./models/cfg/yolov8_sa_innerciou01.yaml",
            "./models/cfg/yolov8_sa_innerciou02.yaml",
            "./models/cfg/yolov8_sa_fix.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou01.yaml",
            "./models/cfg/yolov8_sa_fix_innerciou02.yaml",
            "./models/cfg/yolov8_sppelan.yaml",
            "./models/cfg/yolov8_sppelan_innerciou01.yaml",
            "./models/cfg/yolov8_sppelan_innerciou02.yaml",
            "./models/cfg/yolov8_sa_sppelan.yaml",
            "./models/cfg/yolov8_sa_sppelan_innerciou01.yaml",
            "./models/cfg/yolov8_sa_sppelan_innerciou02.yaml",
            "./models/cfg/yolov8_sa_fix_sppelan.yaml",
            "./models/cfg/yolov8_sa_fix_sppelan_innerciou01.yaml",
            "./models/cfg/yolov8_sa_fix_sppelan_innerciou02.yaml",
            "./models/cfg/yolov8_sppfcspc.yaml",
            "./models/cfg/yolov8_sppfcspc_innerciou01.yaml",
            "./models/cfg/yolov8_sppfcspc_innerciou02.yaml",
            "./models/cfg/yolov8_sa_sppfcspc.yaml",
            "./models/cfg/yolov8_sa_sppfcspc_innerciou01.yaml",
            "./models/cfg/yolov8_sa_sppfcspc_innerciou02.yaml",
            "./models/cfg/yolov8_sa_fix_sppfcspc.yaml",
            "./models/cfg/yolov8_sa_fix_sppfcspc_innerciou01.yaml",
            "./models/cfg/yolov8_sa_fix_sppfcspc_innerciou02.yaml",
        ]
    )

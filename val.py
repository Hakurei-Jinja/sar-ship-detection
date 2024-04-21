from src.nn.model import MyYOLO


if __name__ == "__main__":
    model = MyYOLO("./models/trained/train2/weights/best.pt", verbose=True)

    metrics = model.val(data="./datasets/SSDD/cfg/ssdd_all.yaml")
    metrics = model.val(data="./datasets/SSDD/cfg/ssdd_inshore.yaml")
    metrics = model.val(data="./datasets/SSDD/cfg/ssdd_offshore.yaml")

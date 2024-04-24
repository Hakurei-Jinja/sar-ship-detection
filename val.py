from src.nn.model import MyYOLO


def detect_val(path: str):
    model = MyYOLO(path, verbose=True)
    model.val(data="./datasets/SSDD/cfg/detect/ssdd_all.yaml")
    model.val(data="./datasets/SSDD/cfg/detect/ssdd_inshore.yaml")
    model.val(data="./datasets/SSDD/cfg/detect/ssdd_offshore.yaml")


def obb_val(path: str):
    model = MyYOLO(path, verbose=True)
    model.val(data="./datasets/SSDD/cfg/obb/ssdd_all_obb.yaml")
    model.val(data="./datasets/SSDD/cfg/obb/ssdd_inshore_obb.yaml")
    model.val(data="./datasets/SSDD/cfg/obb/ssdd_offshore_obb.yaml")


def seg_val(path: str):
    model = MyYOLO(path, verbose=True)
    model.val(data="./datasets/SSDD/cfg/seg/ssdd_all_seg.yaml")
    model.val(data="./datasets/SSDD/cfg/seg/ssdd_inshore_seg.yaml")
    model.val(data="./datasets/SSDD/cfg/seg/ssdd_offshore_seg.yaml")


if __name__ == "__main__":
    detect_val("./models/pt/detect/v8n/weights/best.pt")
    # detect_val("./models/pt/detect/v8n-sa1/weights/best.pt")

    # obb_val("./models/pt/obb/v8n/weights/best.pt")
    # obb_val("./models/pt/obb/v8n-sa1/weights/best.pt")
    # obb_val("./models/pt/obb/v8n-sa2/weights/best.pt")

    # seg_val("./models/pt/seg/v8n/weights/best.pt")
    # seg_val("./models/pt/seg/v8n-sa1/weights/best.pt")
    # seg_val("./models/pt/seg/v8n-sa2/weights/best.pt")

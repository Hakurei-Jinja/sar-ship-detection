from src.nn.model import MyYOLO


def ssdd_detect_val(path: str):
    model = MyYOLO(path, verbose=True)
    model.val(data="./datasets/SSDD/cfg/detect/ssdd_all.yaml", plots=True)
    model.val(data="./datasets/SSDD/cfg/detect/ssdd_inshore.yaml", plots=True)
    model.val(data="./datasets/SSDD/cfg/detect/ssdd_offshore.yaml", plots=True)


def ssdd_obb_val(path: str):
    model = MyYOLO(path, verbose=True)
    model.val(data="./datasets/SSDD/cfg/obb/ssdd_all_obb.yaml", plots=True)
    model.val(data="./datasets/SSDD/cfg/obb/ssdd_inshore_obb.yaml", plots=True)
    model.val(data="./datasets/SSDD/cfg/obb/ssdd_offshore_obb.yaml", plots=True)


if __name__ == "__main__":
    ssdd_prefix = "./models/pt/ssdd"
    ssdd_detect_val(ssdd_prefix + "/detect/v8n/weights/best.pt")
    ssdd_detect_val(ssdd_prefix + "/detect/v8n-dc/weights/best.pt")
    ssdd_detect_val(ssdd_prefix + "/detect/v8n-sh/weights/best.pt")
    ssdd_detect_val(ssdd_prefix + "/detect/v8n-sh-dc/weights/best.pt")
    ssdd_detect_val(ssdd_prefix + "/detect/v8s/weights/best.pt")
    ssdd_obb_val(ssdd_prefix + "/obb/v8n/weights/best.pt")
    ssdd_obb_val(ssdd_prefix + "/obb/v8n-dc/weights/best.pt")
    ssdd_obb_val(ssdd_prefix + "/obb/v8n-sh/weights/best.pt")
    ssdd_obb_val(ssdd_prefix + "/obb/v8n-sh-dc/weights/best.pt")
    ssdd_obb_val(ssdd_prefix + "/obb/v8s/weights/best.pt")

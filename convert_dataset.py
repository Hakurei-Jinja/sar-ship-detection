from src.dataset import DatasetConvertor


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
    convert_SSDD()

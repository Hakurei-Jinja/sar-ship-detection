from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    name: str
    path: str
    structure_img_path: str
    train_img_path: str
    eval_img_path: str

    __getitem__ = lambda self, key: getattr(self, key)
    __setitem__ = lambda self, key, value: setattr(self, key, value)
    dict = asdict

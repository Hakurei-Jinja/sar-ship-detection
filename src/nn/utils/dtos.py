from dataclasses import dataclass


@dataclass
class ModelConfig:
    depth: float
    width: float
    max_channels: float
    nc: int
    activation: str | None
    kpt_shape: float

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclass
class LayerConfig:
    former_ch: list[int]
    from_index: int | list[int]
    repeat_num: int
    args: list

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

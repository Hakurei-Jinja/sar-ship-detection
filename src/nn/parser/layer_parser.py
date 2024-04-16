from abc import ABCMeta, abstractmethod

import torch.nn as nn
from ultralytics.utils.torch_utils import deepcopy, make_divisible

from ..modules import *
from ..utils import LayerConfig, ModelConfig


class LayerParser(metaclass=ABCMeta):
    @abstractmethod
    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        pass

    @abstractmethod
    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        pass

    @abstractmethod
    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        pass


class DefaultLayerParser(LayerParser):
    _module_cls: type

    def __init__(self, module_cls: type):
        self._module_cls = module_cls

    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        return self._repeat_module(
            self._module_cls(*self.get_args(model_cfg, layer_cfg)), layer_cfg.repeat_num
        )

    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        return layer_cfg.args

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        return self._get_from_index_ch(layer_cfg)

    @staticmethod
    def _repeat_module(module: nn.Module, n) -> nn.Sequential | nn.Module:
        return nn.Sequential(*(module for _ in range(n))) if n > 1 else module

    @staticmethod
    def _get_from_index_ch(layer_cfg: LayerConfig) -> int:
        index = layer_cfg.from_index
        if not isinstance(index, int):
            raise ValueError("from_index should be an integer")
        return layer_cfg.former_ch[index]


class NNUpSampleParser(DefaultLayerParser):
    pass


class ConvParser(DefaultLayerParser):
    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        ch_in = self._get_from_index_ch(layer_cfg)
        ch_out = self.get_out_channels(model_cfg, layer_cfg)
        return [ch_in, ch_out, *layer_cfg.args[1:]]

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        ch_out = layer_cfg.args[0]
        return make_divisible(min(ch_out, model_cfg.max_channels) * model_cfg.width, 8)


class SPPFParser(ConvParser):
    pass


class ASPPParser(ConvParser):
    pass


class RepNCSPELAN4Parser(ConvParser):
    pass


class ADownParser(ConvParser):
    pass


class SPPELANParser(DefaultLayerParser):
    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        ch_in = self._get_from_index_ch(layer_cfg)
        ch_out = self.get_out_channels(model_cfg, layer_cfg)
        ch_mid = layer_cfg.args[1]
        ch_mid = make_divisible(
            min(ch_mid, model_cfg.max_channels) * model_cfg.width, 8
        )
        return [ch_in, ch_out, ch_mid, *layer_cfg.args[2:]]

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        ch_out = layer_cfg.args[0]
        return make_divisible(min(ch_out, model_cfg.max_channels) * model_cfg.width, 8)


class C2fParser(DefaultLayerParser):
    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        return self._module_cls(*self.get_args(model_cfg, layer_cfg))

    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        ch_in = self._get_from_index_ch(layer_cfg)
        ch_out = self.get_out_channels(model_cfg, layer_cfg)
        repeat_num = layer_cfg.repeat_num
        return [ch_in, ch_out, repeat_num, *layer_cfg.args[1:]]

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        ch_out = layer_cfg.args[0]
        return make_divisible(min(ch_out, model_cfg.max_channels) * model_cfg.width, 8)


class ShuffleAttentionParser(DefaultLayerParser):
    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        groups = make_divisible(layer_cfg.args[0] * model_cfg.width, 8)
        return [self._get_from_index_ch(layer_cfg), groups]


class ConcatParser(DefaultLayerParser):
    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        from_index = layer_cfg.from_index
        if isinstance(from_index, int):
            from_index = [from_index]
        return sum([layer_cfg.former_ch[i] for i in from_index])


class DetectParser(DefaultLayerParser):
    def get_args(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> list:
        args = deepcopy(layer_cfg.args)
        from_index = layer_cfg.from_index
        if isinstance(from_index, int):
            from_index = [from_index]
        args.append([layer_cfg.former_ch[i] for i in from_index])
        return args

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        return 0


class LayerParserFactory:
    @staticmethod
    def get_parser(module_cls: type) -> LayerParser:
        if module_cls is Conv:
            return ConvParser(module_cls)
        elif module_cls is ADown:
            return ADownParser(module_cls)
        elif module_cls is ShuffleAttention:
            return ShuffleAttentionParser(module_cls)
        elif module_cls is SPPF:
            return SPPFParser(module_cls)
        elif module_cls is SPPELAN:
            return SPPELANParser(module_cls)
        elif module_cls is ASPP:
            return ASPPParser(module_cls)
        elif module_cls is C2f:
            return C2fParser(module_cls)
        elif module_cls is RepNCSPELAN4:
            return RepNCSPELAN4Parser(module_cls)
        elif module_cls is Concat:
            return ConcatParser(module_cls)
        elif module_cls is nn.Upsample:
            return NNUpSampleParser(module_cls)
        elif module_cls is Detect:
            return DetectParser(module_cls)
        raise ValueError(f"Unsupported module class: {module_cls}")

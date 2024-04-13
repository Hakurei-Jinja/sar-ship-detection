from abc import ABCMeta, abstractmethod

import torch.nn as nn
from ultralytics.utils.torch_utils import make_divisible

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


class DefaultLayerParser(LayerParser):
    _module_cls: type

    def __init__(self, module_cls: type):
        self._module_cls = module_cls

    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        return self._repeat_module(
            self._module_cls(*layer_cfg.args), layer_cfg.repeat_num
        )

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        return layer_cfg.former_ch[-1]

    @staticmethod
    def _repeat_module(module: nn.Module, n) -> nn.Sequential | nn.Module:
        return nn.Sequential(*(module for _ in range(n))) if n > 1 else module


class NNUpSampleParser(DefaultLayerParser):
    pass


class ConvParser(DefaultLayerParser):
    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        if model_cfg.activation:
            self._module_cls.default_act = model_cfg.activation
        ch_in = layer_cfg.former_ch[-1]
        ch_out = self.get_out_channels(model_cfg, layer_cfg)
        args = [ch_in, ch_out, *layer_cfg.args[1:]]
        return self._repeat_module(self._module_cls(*args), layer_cfg.repeat_num)

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        ch_out = layer_cfg.args[0]
        return make_divisible(min(ch_out, model_cfg.max_channels) * model_cfg.width, 8)


class SPPFParser(ConvParser):
    pass


class SPPELANParser(ConvParser):
    pass


class RepNCSPELAN4Parser(ConvParser):
    pass


class ADownParser(ConvParser):
    pass


class C2fParser(DefaultLayerParser):
    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        ch_in = layer_cfg.former_ch[-1]
        ch_out = self.get_out_channels(model_cfg, layer_cfg)
        repeat_num = layer_cfg.repeat_num
        args = [ch_in, ch_out, repeat_num, *layer_cfg.args[1:]]
        return self._module_cls(*args)

    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        ch_out = layer_cfg.args[0]
        return make_divisible(min(ch_out, model_cfg.max_channels) * model_cfg.width, 8)


class ConcatParser(DefaultLayerParser):
    def get_out_channels(self, model_cfg: ModelConfig, layer_cfg: LayerConfig) -> int:
        from_index = layer_cfg.from_index
        if isinstance(from_index, int):
            from_index = [from_index]
        return sum([layer_cfg.former_ch[i] for i in from_index])


class DetectParser(DefaultLayerParser):
    def get_module_layer(
        self, model_cfg: ModelConfig, layer_cfg: LayerConfig
    ) -> nn.Sequential | nn.Module:
        args = layer_cfg.args
        from_index = layer_cfg.from_index
        if isinstance(from_index, int):
            from_index = [from_index]
        args.append([layer_cfg.former_ch[i] for i in from_index])
        return self._module_cls(*layer_cfg.args)


class LayerParserFactory:
    @staticmethod
    def get_parser(module_cls: type) -> LayerParser:
        if module_cls is Conv:
            return ConvParser(module_cls)
        elif module_cls is RepNCSPELAN4:
            return RepNCSPELAN4Parser(module_cls)
        elif module_cls is ADown:
            return ADownParser(module_cls)
        elif module_cls is SPPF:
            return SPPFParser(module_cls)
        elif module_cls is SPPELAN:
            return SPPELANParser(module_cls)
        elif module_cls is C2f:
            return C2fParser(module_cls)
        elif module_cls is nn.Upsample:
            return NNUpSampleParser(module_cls)
        elif module_cls is Concat:
            return ConcatParser(module_cls)
        elif module_cls is Detect:
            return DetectParser(module_cls)
        raise ValueError(f"Unsupported module class: {module_cls}")

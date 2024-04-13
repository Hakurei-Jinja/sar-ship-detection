import ast
import contextlib
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from ultralytics.utils import colorstr

from ..modules import *
from ..utils import LayerConfig, Logger, ModelConfig
from .layer_parser import LayerParserFactory


@dataclass
class LayerParams:
    name: str
    index: int
    from_index: int | list[int]


class ModelParser:
    __classes = globals().copy()
    __logger: Logger = Logger(True)

    def __log_layer(self, model_param: dict):
        index = model_param.get("index")
        f = model_param.get("from")
        n = model_param.get("repeat")
        p = model_param.get("params")
        m = model_param.get("module")
        args = model_param.get("arguments")
        self.__logger.log(f"{index:>3}{f:>20}{n:>3}{p:10.0f}  {m:<45}{str(args):<30}")

    def __log_title(self):
        self.__logger.log(
            f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )

    def __log_activation(self, model_cfg: ModelConfig):
        if model_cfg.activation:
            self.__logger.log(f"{colorstr('activation:')} {model_cfg.activation}")

    def parse_model(self, cfg, ch, verbose):
        """
        Parse the model from the given config.

        Args:
            cfg (dict): The model config.
            ch (int): The number of input channels.
            verbose (bool): Whether to print the model info.

        Returns:
            tuple[nn.Sequential, list[int]]: The model and the save list.
        """
        self.__logger = Logger(verbose)
        model_cfg = self.__get_cfg(cfg)
        self.__log_activation(model_cfg)
        self.__log_title()

        former_ch, layers, savelist = [], [], []
        for i, (from_index, repeat_num, cls_name, args) in enumerate(
            cfg["backbone"] + cfg["head"]
        ):
            module = self.__get_module_cls(cls_name)
            repeat_num = self.__calculate_repeat_num(repeat_num, model_cfg.depth)
            args = [self.__convert_arg(arg, model_cfg) for arg in args]
            layer_config = self.__rewrite_layer_cfg_by_index(
                i, ch, LayerConfig(former_ch, from_index, repeat_num, args)
            )

            layer_parser = LayerParserFactory.get_parser(module)
            layer = layer_parser.get_module_layer(model_cfg, layer_config)
            layer_name = self.__get_layer_name(module)
            layer = self.__set_layer_params(
                LayerParams(name=layer_name, index=i, from_index=from_index), layer
            )
            former_ch.append(layer_parser.get_out_channels(model_cfg, layer_config))
            layers.append(layer)
            savelist.extend(self.__get_save_list(from_index, i))
            self.__log_layer(
                {
                    "index": i,
                    "from": str(from_index),
                    "repeat": repeat_num,
                    "params": layer.np,
                    "module": layer_name,
                    "arguments": str(args),
                }
            )
        return nn.Sequential(*layers), sorted(savelist)

    def __get_cfg(self, cfg: dict) -> ModelConfig:
        nc = cfg.get("nc")
        if not nc:
            raise ValueError("nc not defined")
        activation = cfg.get("activation")
        kpt_shape = cfg.get("kpt_shape", 1.0)
        depth, width, max_channels = self.__get_cfg_params(cfg)
        return ModelConfig(
            depth=depth,
            width=width,
            max_channels=max_channels,
            nc=nc,
            activation=activation,
            kpt_shape=kpt_shape,
        )

    @staticmethod
    def __get_module_cls(cls_name: str) -> type:
        return (
            getattr(nn, cls_name[3:])
            if "nn." in cls_name
            else ModelParser.__classes[cls_name]
        )

    @staticmethod
    def __calculate_repeat_num(repeat_num: int, depth: float) -> int:
        return max(round(repeat_num * depth), 1) if repeat_num > 1 else repeat_num

    @staticmethod
    def __convert_arg(arg, model_cfg: ModelConfig) -> Any:
        if not isinstance(arg, str):
            return arg
        with contextlib.suppress(ValueError):
            arg = (
                model_cfg[arg]
                if arg in model_cfg.__annotations__
                else ast.literal_eval(arg)
            )
        return arg

    @staticmethod
    def __rewrite_layer_cfg_by_index(
        index: int, input_ch: int, layer_cfg: LayerConfig
    ) -> LayerConfig:
        if index != 0:
            return layer_cfg
        return LayerConfig(
            former_ch=[input_ch],
            from_index=layer_cfg.from_index,
            repeat_num=layer_cfg.repeat_num,
            args=layer_cfg.args,
        )

    @staticmethod
    def __get_layer_name(module: type) -> str:
        return str(module)[8:-2].replace("__main__.", "")

    @staticmethod
    def __set_layer_params(
        params: LayerParams, layer: nn.Module | nn.Sequential
    ) -> nn.Module | nn.Sequential:
        layer.i, layer.f, layer.type = (params.index, params.from_index, params.name)  # type: ignore
        layer.np = sum(x.numel() for x in layer.parameters())  # type: ignore
        return layer

    @staticmethod
    def __get_save_list(from_index: int | list[int], index: int) -> list[int]:
        return [
            x % index
            for x in ([from_index] if isinstance(from_index, int) else from_index)
            if x != -1
        ]

    def __get_cfg_params(self, cfg: dict) -> tuple[float, float, float]:
        scales = cfg.get("scales")
        if scales:
            return self.__get_scale_params(scales, cfg.get("scale"))
        return self.__get_default_params(cfg)

    def __get_scale_params(
        self, scales: dict, scale: str | None
    ) -> tuple[float, float, float]:
        if not scale:
            scale = tuple(scales.keys())[0]
            self.__logger.warn(
                f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'."
            )
        return scales[scale]

    @staticmethod
    def __get_default_params(cfg: dict) -> tuple[float, float, float]:
        return (
            cfg.get("depth_multiple", 1.0),
            cfg.get("width_multiple", 1.0),
            float("inf"),
        )

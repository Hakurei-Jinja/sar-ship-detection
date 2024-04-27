from torchvision.ops import DeformConv2d
import torch
from torch import nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class DeformConv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = DeformConvV3(c1, c2, k, s, autopad(k, p, d), d, g, bias=False)  # type: ignore
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DeformConvV1(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int],
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] = 0,
        d: int | tuple[int, int] = 1,
        g: int = 1,
        offset_g: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.offset_conv = nn.Conv2d(c1, self.__get_offset_c2(k, offset_g), k, s, p, d)
        self.deform_conv = DeformConv2d(c1, c2, k, s, p, d, g, bias)  # type: ignore
        self.__init_weights()

    def forward(self, x):
        return self.deform_conv(x, self.offset_conv(x))

    def __init_weights(self):
        self.offset_conv.weight = nn.init.zeros_(self.offset_conv.weight)

    @staticmethod
    def __get_offset_c2(k: int | tuple[int, int], g: int) -> int:
        kh, kw = DeformConvV1.__get_kernel_size(k)
        return 2 * kh * kw * g

    @staticmethod
    def __get_kernel_size(k) -> tuple:
        return k if isinstance(k, tuple) else (k, k)


class DeformConvV2(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int],
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] = 0,
        d: int | tuple[int, int] = 1,
        g: int = 1,
        offset_g: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.offset_conv = nn.Conv2d(c1, self.__get_offset_c2(k, offset_g), k, s, p, d)
        self.mask_conv = nn.Conv2d(c1, self.__get_mask_c2(k, offset_g), k, s, p, d)
        self.deform_conv = DeformConv2d(c1, c2, k, s, p, d, g, bias)  # type: ignore
        self.__init_weights()

    def forward(self, x):
        mask = torch.sigmoid(self.mask_conv(x))
        return self.deform_conv(x, self.offset_conv(x), mask)

    def __init_weights(self):
        self.offset_conv.weight = nn.init.zeros_(self.offset_conv.weight)

    @staticmethod
    def __get_offset_c2(k: int | tuple[int, int], g: int) -> int:
        kh, kw = DeformConvV2.__get_kernel_size(k)
        return 2 * kh * kw * g

    @staticmethod
    def __get_mask_c2(k: int | tuple[int, int], g: int) -> int:
        kh, kw = DeformConvV2.__get_kernel_size(k)
        return kh * kw * g

    @staticmethod
    def __get_kernel_size(k) -> tuple:
        return k if isinstance(k, tuple) else (k, k)


class DeformConvV3(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, int],
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] = 0,
        d: int | tuple[int, int] = 1,
        g: int = 1,
        offset_g: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        offset_c2 = self.__get_offset_c2(k, offset_g)
        mask_c2 = self.__get_mask_c2(k, offset_g)
        self.offset_conv = nn.Conv2d(c1, offset_c2, k, s, p, d)
        self.mask_conv = nn.Conv2d(c1, mask_c2, k, s, p, d)
        self.deform_conv = DeformConv2d(c1, c2, k, s, p, d, g, bias)  # type: ignore
        self.conv1x1 = nn.Conv2d(c2 + offset_c2 + mask_c2, c2, 1)
        self.__init_weights()

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        x = self.deform_conv(x, offset, mask)
        return self.conv1x1(torch.cat([x, offset, mask], dim=1))

    def __init_weights(self):
        self.offset_conv.weight = nn.init.zeros_(self.offset_conv.weight)

    @staticmethod
    def __get_offset_c2(k: int | tuple[int, int], g: int) -> int:
        kh, kw = DeformConvV3.__get_kernel_size(k)
        return 2 * kh * kw * g

    @staticmethod
    def __get_mask_c2(k: int | tuple[int, int], g: int) -> int:
        kh, kw = DeformConvV3.__get_kernel_size(k)
        return kh * kw * g

    @staticmethod
    def __get_kernel_size(k) -> tuple:
        return k if isinstance(k, tuple) else (k, k)

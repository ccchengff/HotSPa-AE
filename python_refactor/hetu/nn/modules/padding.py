import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair, _quadruple

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'ConstantPad2d',
]

class ConstantPadNd(Module):

    def __init__(self, value: float) -> None:
        with hetu.graph("define_and_run"):
            super(ConstantPadNd, self).__init__()
            self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return hetu.pad(input, self.padding, "constant", self.value)

class ConstantPad2d(ConstantPadNd):
    def __init__(self, padding: Tuple[int, int, int, int], value: float) -> None:
        with hetu.graph("define_and_run"):
            super(ConstantPad2d, self).__init__(value)
            self.padding = _quadruple(padding)


class ZeroPad2d(ConstantPad2d):
    def __init__(self, padding: Tuple[int, int, int, int]) -> None:
        with hetu.graph("define_and_run"):
            super(ZeroPad2d, self).__init__(padding, 0.)
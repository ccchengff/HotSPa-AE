import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'MaxPoolNd', 
    'MaxPool2d', 
    'AvgPoolNd', 
    'AvgPool2d', 
]

class MaxPoolNd(Module):

    def __init__(self, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...]) -> None:
        with hetu.graph("define_and_run"):
            super(MaxPoolNd, self).__init__()
            self.kernel_size = list(kernel_size)
            self.stride = list(stride)
            self.padding = list(padding)


class MaxPool2d(MaxPoolNd):

    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0) -> None:
        with hetu.graph("define_and_run"):
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = _pair(padding)
            super(MaxPool2d, self).__init__(
                kernel_size_, stride_, padding_)

    def forward(self, input: Tensor) -> Tensor:
        return hetu.maxpool(input, self.kernel_size[0], self.kernel_size[1], self.padding[0], self.stride[0])


class AvgPoolNd(Module):

    def __init__(self, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...]) -> None:
        with hetu.graph("define_and_run"):
            super(AvgPoolNd, self).__init__()
            self.kernel_size = list(kernel_size)
            self.stride = list(stride)
            self.padding = list(padding)


class AvgPool2d(AvgPoolNd):

    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0) -> None:
        with hetu.graph("define_and_run"):
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = _pair(padding)
            super(AvgPool2d, self).__init__(
                kernel_size_, stride_, padding_)

    def forward(self, input: Tensor) -> Tensor:
        return hetu.avgpool(input, self.kernel_size[0], self.kernel_size[1], self.padding[0], self.stride[0])
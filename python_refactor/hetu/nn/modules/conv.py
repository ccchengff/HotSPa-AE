import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [ 
    'Conv2d', 
]

class ConvNd(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...], padding: Tuple[int, ...], bias: bool) -> None:
        with hetu.graph("define_and_run"):
            super(ConvNd, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = list(kernel_size)
            self.stride = list(stride)
            self.padding = list(padding)
            self.weight = hetu.nn.functional.kaiming_uniform_([out_channels, in_channels, *kernel_size], 
                                                              a = math.sqrt(5), requires_grad=True)
            if bias:
                fan_in, _ = hetu.nn.functional._calculate_fan_in_and_fan_out(self.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                self.bias = hetu.rand([out_channels], -bound, bound, requires_grad=True)
            else:
                self.bias = None
                # self.register_parameter('bias', None)
            # self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = hetu.nn.init.kaiming_uniform_(a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = hetu.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            hetu.nn.init.uniform_(self.bias, -bound, bound)



class Conv2d(ConvNd):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, bias: bool = True) -> None:
        with hetu.graph("define_and_run"):
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = _pair(padding)
            super(Conv2d, self).__init__(
                in_channels, out_channels, kernel_size_, stride_, padding_, bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return hetu.conv2d(input, self.weight, self.padding[0], self.stride[0])
        else:
            return hetu.conv2d(input, self.weight, self.bias, self.padding[0], self.stride[0])
        
    def to(self, dtype):
        with hetu.graph("define_and_run"):
            self.weight = self.weight.to(datatype = dtype)
            if (self.bias is not None):
                self.bias = self.bias.to(datatype = dtype)

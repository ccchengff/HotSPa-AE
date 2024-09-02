import hetu
from hetu import Tensor
from .module import Module
import math

from typing import Any

__all__ = [
    'Identity', 
    'Linear', 
]

class Identity(Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

class Linear(Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device_group = None) -> None:
        with hetu.graph("define_and_run"):
            super(Linear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = hetu.nn.functional.kaiming_uniform_([out_features, in_features], a = math.sqrt(5), requires_grad=True, device_group = device_group)
            if bias:
                fan_in, _ = hetu.nn.functional._calculate_fan_in_and_fan_out(self.weight.shape)
                bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias = hetu.rand([out_features], -bound, bound, requires_grad=True, device_group = device_group)
            else:
                self.register_parameter('bias', None)
            # self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # hetu.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            pass
            # fan_in, _ = hetu.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
            # hetu.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            return hetu.matmul(input, self.weight, trans_b=True) + self.bias
        else:
            return hetu.matmul(input, self.weight, trans_b=True)

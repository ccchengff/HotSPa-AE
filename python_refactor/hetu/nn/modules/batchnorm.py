import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'NormBase',
    'BatchNorm',
]

class NormBase(Module):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super(NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # self.weight = hetu.nn.Parameter(hetu.ones([num_features], requires_grad=True))
        # self.bias = hetu.nn.Parameter(hetu.zeros([num_features], requires_grad=True))

class BatchNorm(NormBase):
    #TODO:Normalize operators should have only one output.Now we have three. 

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        with hetu.graph("define_and_run"):
            super(BatchNorm, self).__init__(num_features, eps, momentum)
            self.weight = hetu.ones([num_features], requires_grad=True)
            self.bias = hetu.zeros([num_features], requires_grad=True)
            self.running_mean = hetu.empty([num_features], requires_grad=False)
            self.running_var = hetu.empty([num_features], requires_grad=False)
            # self.save_mean = hetu.nn.Parameter(hetu.empty([num_features], requires_grad=False))
            # self.save_var = hetu.nn.Parameter(hetu.empty([num_features], requires_grad=False))

    def forward(self, input: Tensor) -> Tensor:
        # tmp_weight = hetu.nn.Parameter(hetu.ones([self.num_features], requires_grad=True))
        # tmp_bias = hetu.nn.Parameter(hetu.zeros([self.num_features], requires_grad=True))
        return hetu.batch_norm(input, self.weight, self.bias, self.running_mean, self.running_var, self.momentum, self.eps)[0]


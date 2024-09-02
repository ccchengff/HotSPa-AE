import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'Dropout', 
    'Dropout2d', 
]

class DropoutNd(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        with hetu.graph("define_and_run"):
            super(DropoutNd, self).__init__()
            if p < 0 or p > 1:
                raise ValueError("dropout probability has to be between 0 and 1, "
                                "but got {}".format(p))
            self.p = 1 - p
            self.inplace = inplace

class Dropout(DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        # return input
        if self.inplace:
            return hetu.dropout_(input, self.p, False)
        else:
            return hetu.dropout(input, self.p, False)

class Dropout2d(DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if self.inplace:
            return hetu.dropout2d_(input, self.p, False)
        else:
            return hetu.dropout2d(input, self.p, False)
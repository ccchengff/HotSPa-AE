import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair, _quadruple

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'NLLLoss', 'KLDivLoss',
    'MSELoss', 'BCELoss',
]

class _Loss(Module):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        with hetu.graph("define_and_run"):
            super(_Loss, self).__init__()
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        with hetu.graph("define_and_run"):
            super(_WeightedLoss, self).__init__(reduction)
            self.register_buffer('weight', weight)
            self.weight: Optional[Tensor]


class NLLLoss(_WeightedLoss):
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        with hetu.graph("define_and_run"):
            super(NLLLoss, self).__init__(weight, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return hetu.nll_loss(input, target)

class BCELoss(_WeightedLoss):
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        with hetu.graph("define_and_run"):
            super(BCELoss, self).__init__(weight, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return hetu.binary_cross_entropy(input, target)

class KLDivLoss(_WeightedLoss):
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        with hetu.graph("define_and_run"):
            super(KLDivLoss, self).__init__(weight, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return hetu.kl_div(input, target)

class MSELoss(_WeightedLoss):
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        with hetu.graph("define_and_run"):
            super(MSELoss, self).__init__(weight, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return hetu.mse_loss(input, target)
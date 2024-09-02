import hetu
from hetu import Tensor
from .optimizer import Optimizer
from typing import Optional, Iterable

class SGD(Optimizer):

    def __init__(self, params: Optional[Iterable[Tensor]] = None, 
                 lr: float = 0.1, momentum: float = 0.0, nesterov: bool = False):
        # if not isinstance(lr, float):
        #     lr = float(lr)
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1:
            raise ValueError(f"Invalid momemtum: {momentum}")
        if nesterov and momentum == 0:
            raise ValueError(f"Nesterov requires non-zero momentum")
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)
    
    def apply_dense(self, param, grad):
        return hetu.sgd_update(param, grad, self.defaults["lr"])

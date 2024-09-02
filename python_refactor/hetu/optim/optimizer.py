import hetu
from hetu import Tensor
from typing import Optional, Iterable

class Optimizer(object):

    def __init__(self, params: Optional[Iterable[Tensor]], defaults: dict):
        self.params = params
        if self.params is not None:
            self.params = list(self.params)
            assert len(self.params) > 0, "No variables are provided"
            for p in self.params:
                print(type(p), p.requires_grad)
                assert p.requires_grad, f"Parameter {p} is not requires_grad"
        
        self.defaults = defaults
    
    def _assert_dar_mode(self, fn):
        if self.params is not None:
            raise ValueError(
                f"Function '{fn}' should only be used in define-and-run mode")
    
    def _assert_dbr_mode(self, fn):
        if self.params is None:
            raise ValueError(
                f"Function '{fn}' should only be used in define-by-run mode")
    
    def zero_grad(self):
        self._assert_dbr_mode('zero_grad')
        for p in self.params:
            p.zero_grad()
    
    def step(self):
        self._assert_dbr_mode('step')
        update_ops = []
        for p in self.params:
            if p.grad is None:
                continue
            update_op = self.apply_dense(p, p.grad)
            update_ops.append(update_op)
        hetu.group(update_ops).get_or_compute()
    
    def minimize(self, loss, var_list=None, name=None, grad_loss=None):
        # self._assert_dar_mode('minimize')
        update_ops = []
        grads_and_vars = self.compute_gradients(
            loss, 
            var_list=var_list, 
            grad_loss=grad_loss)
        for g, v in grads_and_vars:
            if g is None:
                continue
            update_op = self.apply_dense(v, g)
            update_ops.append(update_op)
        return hetu.group(update_ops, name=name)
    
    def compute_gradients(self, loss, var_list=None, grad_loss=None):
        if var_list is None:
            var_list = hetu.requires_grad_variables(loss)
        else:
            for v in var_list:
                assert v.requires_grad, f"Variable {v} is not requires_grad"
        # for v in var_list:
        #     print(v, " ", v.requires_grad)
        grad_list = hetu.gradients(loss, var_list, grad_loss)
        assert len(grad_list) == len(var_list), \
            f"Only {len(grad_list)} gradients are returned for " + \
            f"{len(var_list)} variables."
        return list(zip(grad_list, var_list))
    
    def apply_dense(self, param, grad):
        raise NotImplementedError

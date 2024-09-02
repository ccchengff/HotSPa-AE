import hetu
from hetu import Tensor
import math

from typing import Tuple, List

def generalized_xavier_(tshape: List, dist: str, mode: str, gain: float, requires_grad: bool = False, dtype = hetu.float32, device_group = None) -> Tensor:
    factor = _calculate_correct_fan(tshape, mode)
    if dist == 'uniform':
        limit = math.sqrt(gain / factor)
        return hetu.rand(tshape, -limit, limit, requires_grad = requires_grad, dtype = dtype, device_group = device_group)
    elif dist == 'normal':
        std = math.sqrt(gain / factor)
        return hetu.randn(tshape, 0., std, requires_grad = requires_grad, dtype = dtype, device_group = device_group)
    else:
        raise ValueError(f"Invalid dist: {dist}")

def xavier_uniform_(tshape: List, gain: float = 1., requires_grad: bool = False, device_group = None) -> Tensor:
    return generalized_xavier_(tshape, 'uniform', 'avg', gain * 3, requires_grad, device_group = device_group)

def xavier_normal_(tshape: List, gain: float = 1., requires_grad: bool = False, device_group = None) -> Tensor:
    return generalized_xavier_(tshape, 'normal', 'avg', gain, requires_grad, device_group = device_group)

def kaiming_uniform_(tshape: List, a: float = 0., mode: str = 'fan_in', 
                     nonlinearity: str = 'leaky_relu', requires_grad: bool = False, 
                     dtype = hetu.float32, device_group = None) -> Tensor:
    gain = calculate_gain(nonlinearity, param=a)
    return generalized_xavier_(tshape, 'uniform', 'fan_in', (gain ** 2) * 3, requires_grad, dtype, device_group = device_group)

he_uniform_ = kaiming_uniform_

def kaiming_normal_(tshape: List, a: float = 0., mode: str = 'fan_in', 
                    nonlinearity: str = 'leaky_relu', 
                    requires_grad: bool = False, device_group = None) -> Tensor:
    gain = calculate_gain(nonlinearity, param=a)
    return generalized_xavier_(tshape, 'normal', 'fan_in', gain ** 2, requires_grad, device_group = device_group)

he_normal_ = kaiming_normal_

def lecun_uniform_(tshape: List, gain: float = 1., requires_grad: bool = False, device_group = None) -> Tensor:
    return generalized_xavier_(tshape, 'uniform', 'fan_in', gain * 3, requires_grad, device_group = device_group)

def lecun_normal_(tshape: List, gain: float = 1., requires_grad: bool = False, device_group = None) -> Tensor:
    return generalized_xavier_(tshape, 'normal', 'fan_in', gain, requires_grad, device_group = device_group)

def _calculate_fan_in_and_fan_out(tshape : List) -> Tuple[int, int]:
    ndim = len(tshape)
    assert ndim >= 2, f"Number of dimensions should be at least 2. Got {ndim}"
    shape = tshape
    hw_scale = 1
    for s in shape[2:]:
        hw_scale *= s
    fan_in = hw_scale * shape[1]
    fan_out = hw_scale * shape[0]
    return fan_in, fan_out

def _calculate_correct_fan(tshape: List, mode: str) -> float:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tshape)
    if mode == 'fan_in':
        return float(fan_in)
    elif mode == 'fan_out':
        return float(fan_out)
    elif mode == 'avg':
        return (fan_in + fan_out) * 0.5
    else:
        raise ValueError(f"Invalid mode: {mode}")

def calculate_gain(nonlinearity: str, param=None):
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 
        'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1.
    elif nonlinearity == 'tanh':
        return 5. / 3.
    elif nonlinearity == 'relu':
        return math.sqrt(2.)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif (not isinstance(param, bool)) and \
             (isinstance(param, int) or isinstance(param, float)):
            negative_slope = param
        else:
            raise ValueError(f"Invalid negative_slope: {param}")
        return math.sqrt(2. / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3. / 4.
    else:
        raise ValueError(f"Invalid nonlinearity: {nonlinearity}")

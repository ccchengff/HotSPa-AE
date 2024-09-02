import hetu
from hetu import Tensor
import math

from typing import Tuple

def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> Tensor:
    return tensor.uniform_(a, b)

def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    return tensor.normal_(mean, std)

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., 
                  a: float = -2., b: float = 2.) -> Tensor:
    return tensor.trunc_normal_(mean, std, a, b)

def constant_(tensor: Tensor, val: float) -> Tensor:
    return tensor.fill_(val)

def ones_(tensor: Tensor) -> Tensor:
    return tensor.fill_(1.)

def zeros_(tensor: Tensor) -> Tensor:
    return tensor.zero_()

def generalized_xavier_(tensor: Tensor, dist: str, mode: str, gain: float) -> Tensor:
    factor = _calculate_correct_fan(tensor, mode)
    if dist == 'uniform':
        limit = math.sqrt(gain / factor)
        return uniform_(tensor, -limit, limit)
    elif dist == 'normal':
        std = math.sqrt(gain / factor)
        return normal_(tensor, 0., std)
    else:
        raise ValueError(f"Invalid dist: {dist}")

def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    return generalized_xavier_(tensor, 'uniform', 'avg', gain * 3)

def xavier_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    return generalized_xavier_(tensor, 'normal', 'avg', gain)

def kaiming_uniform_(tensor: Tensor, a: float = 0., mode: str = 'fan_in', 
                     nonlinearity: str = 'leaky_relu') -> Tensor:
    gain = calculate_gain(nonlinearity, param=a)
    return generalized_xavier_(tensor, 'uniform', 'fan_in', (gain ** 2) * 3)

he_uniform_ = kaiming_uniform_

def kaiming_normal_(tensor: Tensor, a: float = 0., mode: str = 'fan_in', 
                    nonlinearity: str = 'leaky_relu') -> Tensor:
    gain = calculate_gain(nonlinearity, param=a)
    return generalized_xavier_(tensor, 'normal', 'fan_in', gain ** 2)

he_normal_ = kaiming_normal_

def lecun_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    return generalized_xavier_(tensor, 'uniform', 'fan_in', gain * 3)

def lecun_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    return generalized_xavier_(tensor, 'normal', 'fan_in', gain)

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    ndim = tensor.ndim
    assert ndim >= 2, f"Number of dimensions should be at least 2. Got {ndim}"
    shape = tensor.shape
    hw_scale = 1
    for s in shape[2:]:
        hw_scale *= s
    fan_in = hw_scale * shape[1]
    fan_out = hw_scale * shape[0]
    return fan_in, fan_out

def _calculate_correct_fan(tensor: Tensor, mode: str) -> float:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
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

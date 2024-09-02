import hetu
from hetu import Tensor

from typing import TypeVar
_param_type_t = TypeVar('T', bound='Parameter')

class Parameter(Tensor):
    
    def __new__(cls, data: Tensor, requires_grad: bool = True):
        # TODO: support None?
        assert data is not None, "Cannot create Parameter from None"
        if type(data) is Tensor or type(data) is Parameter:
            return Tensor._make_subclass(cls, data.to_variable(requires_grad), requires_grad)
        else:
            raise TypeError(
                f"Cannot create Parameter using data of "
                f"type {type(data).__name__}")

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super(Parameter, self).__repr__()



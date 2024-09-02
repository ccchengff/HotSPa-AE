import collections
from itertools import repeat
from typing import List, Dict, Any

def _nlist(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_single = _nlist(1)
_pair = _nlist(2)
_triple = _nlist(3)
_quadruple = _nlist(4)
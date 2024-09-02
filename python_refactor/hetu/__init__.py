# TODO: output the library to `hetu` directory
import _hetu_core
from _hetu_core import *

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
_tensor_type_t = TypeVar('T', bound='Tensor')

from hetu import nn as nn
from hetu import utils as utils
# from hetu import optim as optim

import builtins # bool is resovled as hetu.bool

cur_graph_contexts = []

class _OpContext(object):
    def __init__(self, 
            eager_device=None, 
            device_group_hierarchy=None, 
            stream_index=None, 
            extra_deps=None
        ):
        self.eager_device = eager_device
        self.device_group_hierarchy = device_group_hierarchy
        self.stream_index = stream_index
        self.extra_deps = extra_deps
        if stream_index is not None:
            if ((not isinstance(stream_index, builtins.int)) or 
                isinstance(stream_index, builtins.bool)):
                raise ValueError(
                    "Expected 'int' type for stream index, got "
                    f"'{type(stream_index).__name__}'")
        if extra_deps is not None:
            if not isinstance(extra_deps, (list, tuple)):
                raise ValueError("Extra dependencies should be lists or tuples of tensors")
            for x in extra_deps:
                if not isinstance(x, Tensor):
                    raise ValueError(f"'{type(x).__name__}' object is not a tensor")
            self.extra_deps = [group(extra_deps)]

    def __enter__(self):
        _hetu_core._internal_context.push_op_ctx(
            eager_device=self.eager_device,
            device_group_hierarchy=self.device_group_hierarchy, 
            stream_index=self.stream_index, 
            extra_deps=self.extra_deps
        )
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_op_ctx(
            pop_eager_device=(self.eager_device is not None),
            pop_device_group_hierarchy=(self.device_group_hierarchy is not None), 
            pop_stream_index=(self.stream_index is not None), 
            pop_extra_deps=(self.extra_deps is not None)
        )

def context(eager_device=None, device_group_hierarchy=None, stream_index=None, extra_deps=None):
    return _OpContext(eager_device=eager_device, device_group_hierarchy=device_group_hierarchy, stream_index=stream_index, extra_deps=extra_deps)

def control_dependencies(control_inputs):
    return _OpContext(extra_deps=control_inputs)

class _GraphContext(object):
    def __init__(self, g, create_new=False, prefix="default", num_strategy=-1, tmp=False):
        if tmp == True:
            assert create_new == True, "Temporary graph must be a new graph"
        self.tmp = tmp
        if isinstance(g, Graph):
            assert create_new == False, f"Using type {type(g).__name__} to create a hetu.Graph is not allowed, please use a str instead."
            self.graph = g
        elif isinstance(g, builtins.str):
            if create_new:
                if g == "define_and_run":
                    self.graph = _hetu_core._internal_context.make_new_define_and_run_graph(prefix + '_' + g)
                elif g == "eager":
                    self.graph = _hetu_core._internal_context.make_new_eager_graph(prefix + '_' + g)
                else:
                    raise NotImplementedError("Can only create define_and_run or eager graph for now.")
            else:
                if prefix == "default":
                    if g == "eager":
                        self.graph = _hetu_core._internal_context.get_default_eager_graph()
                    elif g == "define_and_run":
                        self.graph = _hetu_core._internal_context.get_default_define_and_run_graph()
                    elif g == "define_by_run":
                        self.graph = _hetu_core._internal_context.get_default_define_by_run_graph()
                else:
                    self.graph = _hetu_core._internal_context.get_graph(prefix + '_' + g)
        elif isinstance(g, builtins.int):
            assert create_new == False, f"Using type {type(g).__name__} to create a hetu.Graph is not allowed, please use a str instead."
            self.graph = _hetu_core._internal_context.get_graph(g)
        else:
            raise ValueError(f"Cannot parse type '{type(g).__name__}' as hetu.Graph")
        if num_strategy >= 1:
            # print(f'before set num_startegy: {self.graph.num_strategy}')
            self.graph.set_num_strategy(num_strategy)
            # print(f'after set num_startegy: {self.graph.num_strategy}, cur_strategy_id: {self.graph.cur_strategy_id}')

    def __enter__(self):
        if len(cur_graph_contexts) > 0:
            if str(self.graph) != str(cur_graph_contexts[0].graph):
                pass
                # print(f"Caution! You wrap {self.graph} graph in {cur_graph_contexts[0].graph} graph, we just return you the latter!")
            self.wrapped_context = True
            return cur_graph_contexts[0]
        self.wrapped_context = False
        _hetu_core._internal_context.push_graph_ctx(self.graph.id)
        cur_graph_contexts.append(self)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        if self.wrapped_context:
            return
        _hetu_core._internal_context.pop_graph_ctx()
        cur_graph_contexts.remove(self)
        if self.tmp:
            graph_id = self.graph.id
            self.graph = None
            _hetu_core._internal_context.delete_graph(graph_id)

def graph(g, create_new=False, prefix="default", num_strategy=-1, tmp=False):
    return _GraphContext(g, create_new=create_new, prefix=prefix, num_strategy=num_strategy, tmp=tmp)

class _AutocastContext(object):
    def __init__(self, dtype = None):
        if dtype is None:
            self.autocast = _hetu_core._internal_context.get_default_autocast()
        else:
            self.autocast = _hetu_core._internal_context.make_new_autocast(True, dtype)

    def __enter__(self):
        _hetu_core._internal_context.push_autocast_ctx(self.autocast.id)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_autocast_ctx()

def autocast():
    return _AutocastContext()

def autocast(dtype):
    return _AutocastContext(dtype)

class _RunLevel(object):
    def __init__(self, type):
        if type == "update":
            self.type = 0
        elif type == "grad":
            self.type = 1
        elif type == "alloc":
            self.type = 2
        elif type == "topo":
            self.type = 3
        else:
            raise ValueError(f"No level for '{type}' in hetu.Graph")
        
def run_level(type="update"):
    return _RunLevel(type).type

class _MergeStrategyContext(object):
    def __init__(self, target_graph="default", num_strategy=-1):
        self.graph = _hetu_core._internal_context.make_new_define_and_run_graph("tmp_define_and_run")
        self.graph.set_num_strategy(num_strategy)
        if target_graph == "default":
            self.target_graph = _hetu_core._internal_context.get_default_define_and_run_graph()
        else:
            self.target_graph = _hetu_core._internal_context.get_graph(target_graph + '_' + "define_and_run")
        
    def __enter__(self):
        if len(cur_graph_contexts) > 0:
            assert str(self.graph) != str(cur_graph_contexts[0].graph), \
                f"Caution! You wrap {self.tmp_graph} graph in {cur_graph_contexts[0].graph} graph"
        _hetu_core._internal_context.push_graph_ctx(self.graph.id)
        cur_graph_contexts.append(self)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_graph_ctx()
        cur_graph_contexts.remove(self)
        graph_id = self.graph.id
        self.target_graph.merge_strategy(graph_id)
        self.graph = None
        _hetu_core._internal_context.delete_graph(graph_id)
        
def merge_strategy(target_graph="default", num_strategy=-1):
    return _MergeStrategyContext(target_graph=target_graph, num_strategy=num_strategy)
        
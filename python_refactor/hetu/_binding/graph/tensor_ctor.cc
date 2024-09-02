#include "hetu/_binding/graph/tensor_ctor.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

#define _PY_TENSOR_CTOR_COMMON_ARGS                                            \
  "DataType dtype=None, bool requires_grad=false, DistributedStatesHierarchy ds_hierarchy=None, " OP_META_ARGS

inline PyObject* _from_shape_ctor_helper(ParsedPyArgs& parsed_args, 
                                         Initializer&& init, 
                                         size_t arg_offset = 0) {
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  new(&self->tensor) Tensor();
  self->tensor = MakeParameterOp(
    std::move(init), parsed_args.get_int64_list(0),
    parsed_args.get_dtype_or_peek(arg_offset + 1).value_or(kFloat32), 
    parsed_args.get_bool_or_default(arg_offset + 2),
    parsed_args.get_ds_hierarchy_or_empty(arg_offset + 3),     
    parse_op_meta(parsed_args, arg_offset + 4));
  return reinterpret_cast<PyObject*>(self);
}

inline PyObject* _like_tensor_ctor_helper(ParsedPyArgs& parsed_args,
                                          Initializer&& init, 
                                          size_t arg_offset = 0) {
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  new(&self->tensor) Tensor();
  self->tensor = MakeParameterOp(
    std::move(init), parsed_args.get_tensor(0)->shape(),
    parsed_args.get_dtype_or_peek(arg_offset + 1).value_or(kFloat32),
    parsed_args.get_bool_or_default(arg_offset + 2),
    parsed_args.get_ds_hierarchy_or_empty(arg_offset + 3),
    parse_op_meta(parsed_args, arg_offset + 4));
  return reinterpret_cast<PyObject*>(self);
}

/******************************************************
 * Copy Constrcutor
 ******************************************************/

PyObject* TensorCopyCtor(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);

  static PyArgParser parser({
    "Tensor(numpy.array data, " _PY_TENSOR_CTOR_COMMON_ARGS ")", 
    "Tensor(hetu.NDArray data, " _PY_TENSOR_CTOR_COMMON_ARGS ")", 
    "Tensor(PyObject* data, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  // Question: During device placement, the ndarray data will be copied 
  // if it does not match the expected dtype or device. 
  // Can we defer the copy to device placement?
  if (parsed_args.signature_index() == 0) {
    new(&self->tensor) Tensor();
    NDArray data = NDArrayCopyFromNumpyCtor(
      parsed_args.get_py_obj(0), 
      parsed_args.get_dtype_or_peek(1), 
      nullopt);
    self->tensor = MakeParameterOp(data, false, kUndeterminedDataType,
                                  parsed_args.get_bool_or_default(2),
                                  parsed_args.get_ds_hierarchy_or_empty(3),
                                  parse_op_meta(parsed_args, 4));
  } else if (parsed_args.signature_index() == 1) {
    new (&self->tensor) Tensor();
    NDArray data = NDArrayCopyFromNDArrayCtor(
      parsed_args.get_ndarray(0), parsed_args.get_dtype_or_peek(1), nullopt);
    self->tensor = MakeParameterOp(data, false, kUndeterminedDataType,
                                  parsed_args.get_bool_or_default(2),
                                  parsed_args.get_ds_hierarchy_or_empty(3),
                                  parse_op_meta(parsed_args, 4));
  } else if (parsed_args.signature_index() == 2) {
    new (&self->tensor) Tensor();
    NDArray data = NDArrayCopyFromSequenceCtor(
      parsed_args.get_py_obj(0), parsed_args.get_dtype_or_peek(1), nullopt);
    self->tensor = MakeParameterOp(data, false, kUndeterminedDataType,
                                  parsed_args.get_bool_or_default(2),
                                  parsed_args.get_ds_hierarchy_or_empty(3),
                                  parse_op_meta(parsed_args, 4));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

/******************************************************
 * Placeholder Constrcutor
 ******************************************************/

PyObject* PyTensor_placeholder(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "placeholder(DataType dtype, HTShape shape, DistributedStatesHierarchy ds_hierarchy=None, " OP_META_ARGS ")", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  if (parsed_args.signature_index() == 0) {
    new(&self->tensor) Tensor();
    self->tensor =
      MakePlaceholderOp(NDArrayMeta()
                          .set_dtype(parsed_args.get_dtype(0))
                          .set_shape(parsed_args.get_int64_list(1)),
                        parsed_args.get_ds_hierarchy_or_empty(2),
                        parse_op_meta(parsed_args, 3));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  placeholder, 
  (PyCFunction) PyTensor_placeholder, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_parallel_placeholder(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "parallel_placeholder(DataType dtype, HTShape global_shape, DistributedStatesHierarchy ds_hierarchy, " OP_META_ARGS ")", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  if (parsed_args.signature_index() == 0) {
    new(&self->tensor) Tensor();    
    self->tensor =
      MakeParallelPlaceholderOp(NDArrayMeta()
                                .set_dtype(parsed_args.get_dtype(0))
                                .set_shape(parsed_args.get_int64_list(1)),
                              parsed_args.get_ds_hierarchy(2), 
                              parse_op_meta(parsed_args, 3));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  parallel_placeholder, 
  (PyCFunction) PyTensor_parallel_placeholder, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_parallel_parameter(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "parallel_parameter(Initializer init, HTShape global_shape, List[DistributedStatesUnion] ds_hierarchy, \
     List[int] local_idx=[-1], DataType dtype=None, bool requires_grad=false, " OP_META_ARGS ")", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  if (parsed_args.signature_index() == 0) {
    new(&self->tensor) Tensor();    
    self->tensor =
      MakeParallelParameterOp(*(parsed_args.get_initializer(0)),
                              parsed_args.get_int64_list(1),
                              parsed_args.get_ds_hierarchy(2),
                              parsed_args.get_int64_list_or_default(3),
                              parsed_args.get_dtype_or_peek(4).value_or(kFloat32),
                              parsed_args.get_bool_or_default(5),
                              parse_op_meta(parsed_args, 6));                              
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  parallel_parameter, 
  (PyCFunction) PyTensor_parallel_parameter, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);  

/******************************************************
 * Empty Tensors
 ******************************************************/

PyObject* PyTensor_empty(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "empty(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, VoidifiedInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  empty, 
  (PyCFunction) PyTensor_empty, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_empty_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "empty_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, VoidifiedInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  empty_like, 
  (PyCFunction) PyTensor_empty_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Zero Tensors
 ******************************************************/

PyObject* PyTensor_zeros(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "zeros(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, ZerosInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  zeros, 
  (PyCFunction) PyTensor_zeros, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_zeros_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "zeros_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, ZerosInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  zeros_like, 
  (PyCFunction) PyTensor_zeros_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Ones Tensors
 ******************************************************/

PyObject* PyTensor_ones(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "ones(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, OnesInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  ones, 
  (PyCFunction) PyTensor_ones, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_ones_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "ones_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, OnesInitializer());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  ones_like, 
  (PyCFunction) PyTensor_ones_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Full/Filled Tensors
 ******************************************************/

PyObject* PyTensor_full(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "full(HTShape size, double fill_value, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(
      parsed_args, ConstantInitializer(parsed_args.get_float64(1)), 1);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  full, 
  (PyCFunction) PyTensor_full, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_full_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "full_like(Tensor tensor, double fill_value, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(
      parsed_args, ConstantInitializer(parsed_args.get_float64(1)), 1);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  full_like, 
  (PyCFunction) PyTensor_full_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Random (Uniform) Tensors
 ******************************************************/

PyObject* PyTensor_rand(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "rand(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")",
    "rand(HTShape size, double lb, double ub, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, UniformInitializer());
  }
  else if (parsed_args.signature_index() == 1) {
    return _from_shape_ctor_helper(parsed_args, UniformInitializer(parsed_args.get_float64_or_default(1),
                                                                   parsed_args.get_float64_or_default(2)), 2);
  } 
  else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  rand, 
  (PyCFunction) PyTensor_rand, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_rand_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "rand_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")",
    "rand_like(Tensor tensor, double lb, double ub, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, UniformInitializer());
  }
  else if (parsed_args.signature_index() == 1) {
    return _like_tensor_ctor_helper(parsed_args, UniformInitializer(parsed_args.get_float64_or_default(1),
                                                                    parsed_args.get_float64_or_default(2)), 2);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  rand_like, 
  (PyCFunction) PyTensor_rand_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Random (Normal) Tensors
 ******************************************************/

PyObject* PyTensor_randn(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "randn(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")",
    "randn(HTShape size, double mean, double stddev, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, NormalInitializer());
  } else if (parsed_args.signature_index() == 1) {
    return _from_shape_ctor_helper(parsed_args, NormalInitializer(parsed_args.get_float64_or_default(1),
                                                                  parsed_args.get_float64_or_default(2)), 2);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  randn, 
  (PyCFunction) PyTensor_randn, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_randn_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "randn_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")",
    "randn_like(Tensor tensor, double mean, double stddev, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, NormalInitializer());
  } else if (parsed_args.signature_index() == 1) {
    return _like_tensor_ctor_helper(parsed_args, NormalInitializer(parsed_args.get_float64_or_default(1),
                                                                   parsed_args.get_float64_or_default(2)), 2);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  randn_like, 
  (PyCFunction) PyTensor_randn_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

/******************************************************
 * Random (Truncated Normal) Tensors
 ******************************************************/

PyObject* PyTensor_trunc_randn(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "trunc_randn(HTShape size, " _PY_TENSOR_CTOR_COMMON_ARGS ")",
    "trunc_randn(HTShape size, double mean, double stddev, double lb, double ub, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _from_shape_ctor_helper(parsed_args, TruncatedNormalInitializer());
  } else if (parsed_args.signature_index() == 1) {
    return _from_shape_ctor_helper(parsed_args, TruncatedNormalInitializer(parsed_args.get_float64_or_default(1),
                                                                           parsed_args.get_float64_or_default(2),
                                                                           parsed_args.get_float64_or_default(3),
                                                                           parsed_args.get_float64_or_default(4)), 4);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  trunc_randn, 
  (PyCFunction) PyTensor_trunc_randn, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

PyObject* PyTensor_trunc_randn_like(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN  
  static PyArgParser parser({
    "trunc_randn_like(Tensor tensor, " _PY_TENSOR_CTOR_COMMON_ARGS ")",
    "trunc_randn_like(Tensor tensor, double mean, double stddev, double lb, double ub, " _PY_TENSOR_CTOR_COMMON_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return _like_tensor_ctor_helper(parsed_args, TruncatedNormalInitializer());
  } else if (parsed_args.signature_index() == 1) {
    return _like_tensor_ctor_helper(parsed_args, TruncatedNormalInitializer(parsed_args.get_float64_or_default(1),
                                                                           parsed_args.get_float64_or_default(2),
                                                                           parsed_args.get_float64_or_default(3),
                                                                           parsed_args.get_float64_or_default(4)), 4);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_CLASS_METHOD(
  trunc_randn_like, 
  (PyCFunction) PyTensor_trunc_randn_like, 
  METH_VARARGS | METH_KEYWORDS, 
  nullptr);

} // namespace graph
} // namespace hetu

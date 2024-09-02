#include "hetu/_binding/autograd/tensor.h"
#include "hetu/_binding/autograd/tensor_ctor.h"
#include "hetu/_binding/autograd/function_registry.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/autograd/ops/Variable.h"

namespace hetu {
namespace autograd {

PyObject* PyTensor_New(const Tensor& tensor, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined && !tensor.is_defined()) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyNDArray";
    auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
    new(&self->tensor) Tensor();
    self->tensor = tensor;
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

PyObject* PyTensorList_New(const TensorList& tensors,
                           bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  PyObject* ret = PyList_New(tensors.size());
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc list";
  for (size_t i = 0; i < tensors.size(); i++) {
    auto* tensor_obj = PyTensor_New(tensors[i], return_none_if_undefined);
    PyList_SET_ITEM(ret, i, tensor_obj);
  }
  return ret;
  HT_PY_FUNC_END
}

inline PyObject* PyTensor_pynew(PyTypeObject* type, PyObject* args, 
                                PyObject* kwargs) {
  return TensorCopyCtor(type, args, kwargs);
}

PyObject* PyTensor_make_subclass(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "_make_subclass(PyObject* cls, Tensor data, bool trainable=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    PyObject* cls = parsed_args.get_py_obj(0);
    HT_TYPE_ERROR_IF(!PyType_Check(cls))
      << "Expected argument \"cls\" to be a type (got " 
      << Py_TYPE(cls)->tp_name << ")";
    PyTypeObject* cls_type = reinterpret_cast<PyTypeObject*>(cls);
    HT_TYPE_ERROR_IF(!PyType_IsSubtype(cls_type, PyTensor_Type))
      << "Type " << cls_type->tp_name << " is not derived from hetu.Tensor";

    auto* unsafe_self = cls_type->tp_alloc(cls_type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyNDArray";
    // Question: Is the casting safe?
    auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
    new(&self->tensor) Tensor();

    self->tensor = parsed_args.get_tensor(1);
    HT_TYPE_ERROR_IF(!self->tensor->is_variable())
      << "Subclass of hetu.Tensor must be created from a variable. "
      << "Please detach the tensor first.";
    self->tensor->set_trainable(parsed_args.get_bool_or_default(2));
    
    return reinterpret_cast<PyObject*>(self);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

void PyTensor_dealloc(PyTensor* self) {
  (&self->tensor)->~Tensor();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyTensor_str(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  // Compute?
  return PyUnicode_FromString(std::to_string(self->tensor));
  HT_PY_FUNC_END
}

PyObject* PyTensor_repr(PyTensor* self) {
  return PyTensor_str(self);
}

PyObject* PyTensor_id(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->tensor->id());
  HT_PY_FUNC_END
}

PyObject* PyTensor_name(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->tensor->name());
  HT_PY_FUNC_END
}

PyObject* PyTensor_ndim(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->tensor->ndim());
  HT_PY_FUNC_END
}

PyObject* PyTensor_dim(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "dim()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyLong_FromInteger(self->tensor->ndim());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_shape(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyLongList_FromIntegerList(self->tensor->shape());
  HT_PY_FUNC_END
}

PyObject* PyTensor_size(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "size(int dim=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    if (parsed_args.has(0))
      return PyLong_FromInteger(self->tensor->shape(parsed_args.get_int64(0)));
    else
      return PyLongList_FromIntegerList(self->tensor->shape());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_stride(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "stride(int dim=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    if (parsed_args.has(0))
      return PyLong_FromInteger(self->tensor->stride(parsed_args.get_int64(0)));
    else
      return PyLongList_FromIntegerList(self->tensor->stride());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_is_variable(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->tensor->is_variable());
  HT_PY_FUNC_END
}

PyObject* PyTensor_to_variable(PyTensor* self, PyObject* args,
                               PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "to_variable(bool trainable=false, " OP_META_ARGS ")"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensor_New(self->tensor->to_variable(
      parsed_args.get_bool_or_default(0), 
      parse_op_meta(parsed_args, 1)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_is_trainable(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->tensor->is_trainable());
  HT_PY_FUNC_END
}

PyObject* PyTensor_set_trainable(PyTensor* self, PyObject* args,
                                 PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "set_trainable(bool trainable)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->tensor->set_trainable(parsed_args.get_bool(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_data(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyNDArray_New(self->tensor->GetOrCompute());
  HT_PY_FUNC_END
}

PyObject* PyTensor_get_or_compute(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyNDArray_New(self->tensor->GetOrCompute());
  HT_PY_FUNC_END
}

PyObject* PyTensor_grad(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  return PyTensor_New(self->tensor->Gradient());
  HT_PY_FUNC_END
}

PyObject* PyTensor_zero_grad(PyTensor* self) {
  HT_PY_FUNC_BEGIN
  self->tensor->ZeroGrad();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyTensor_backward(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "backward(Tensor gradient=None)", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    self->tensor->Backward(parsed_args.get_tensor_optional(0));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyTensor_trainable_variables(PyTensor* self, PyObject* args,
                                       PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "trainable_variables(Tensor tensor, bool connect_p2p=true, bool skip_computed=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto topo = TopoSort(
      parsed_args.get_tensor(0), 
      parsed_args.get_bool_or_default(1), 
      parsed_args.get_bool_or_default(2));
    TensorList trainable_vars;
    trainable_vars.reserve(topo.size());
    for (auto& op : topo)
      if (is_trainable_op(op))
        trainable_vars.push_back(op->output(0));
    return PyTensorList_New(trainable_vars);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_gradients(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "gradients(Tensor y, List[Tensor] xs, Tensor grad_y=None)", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensorList_New(Gradients(
      parsed_args.get_tensor(0), 
      parsed_args.get_tensor_list(1), 
      parsed_args.get_tensor_optional(2)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyTensor_from_numpy(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyTensor_Type->tp_alloc(PyTensor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyTensor";
  auto* self = reinterpret_cast<PyTensor*>(unsafe_self);
  
  static PyArgParser parser({
    "from_numpy(numpy.array data)"
  });
  auto parsed_args = parser.parse(args, kwargs);

  if (parsed_args.signature_index() == 0) {
    auto* array_obj = parsed_args.get_numpy_array(0);
    new(&self->tensor) Tensor();
    self->tensor = VariableOp(NDArrayFromNumpy(array_obj), false)->output(0);
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyTensor_to_numpy(PyTensor* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "numpy(bool force=false)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    bool force = parsed_args.get_bool_or_default(0);
    return NDArrayToNumpy(self->tensor->GetOrCompute(), force);
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyTensor_properties[] = {
  {PY_GET_SET_DEF_NAME("id"), (getter) PyTensor_id, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("name"), (getter) PyTensor_name, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("ndim"), (getter) PyTensor_ndim, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("shape"), (getter) PyTensor_shape, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("is_variable"), (getter) PyTensor_is_variable, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("trainable"), (getter) PyTensor_is_trainable, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("data"), (getter) PyTensor_data, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("grad"), (getter) PyTensor_grad, nullptr, nullptr, nullptr}, 
  {nullptr}
};

PyTypeObject PyTensor_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Tensor", /* tp_name */
  sizeof(PyTensor), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyTensor_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyTensor_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyTensor_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  nullptr, /* tp_methods */
  nullptr, /* tp_members */
  PyTensor_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyTensor_pynew, /* tp_new */
};
PyTypeObject* PyTensor_Type = &PyTensor_Type_obj;

std::vector<PyMethodDef> InitTensorPyMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"dim", (PyCFunction) PyTensor_dim, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"size", (PyCFunction) PyTensor_shape, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"stride", (PyCFunction) PyTensor_stride, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"to_variable", (PyCFunction) PyTensor_to_variable, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"trainable_", (PyCFunction) PyTensor_set_trainable, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"numpy", (PyCFunction) PyTensor_to_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"get_or_compute", (PyCFunction) PyTensor_get_or_compute, METH_NOARGS, nullptr }, 
    {"zero_grad", (PyCFunction) PyTensor_zero_grad, METH_NOARGS, nullptr }, 
    {"backward", (PyCFunction) PyTensor_backward, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"_make_subclass", (PyCFunction) PyTensor_make_subclass, METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::autograd::get_registered_tensor_methods());
  return ret;
}

std::vector<PyMethodDef> InitTensorPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"from_numpy", (PyCFunction) PyTensor_from_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"trainable_variables", (PyCFunction) PyTensor_trainable_variables, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"gradients", (PyCFunction) PyTensor_gradients, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::autograd::get_registered_tensor_class_methods());
  return ret;
}

void AddPyTensorTypeToModule(py::module_& module) {
  PyTensor_Type->tp_as_number = &(get_registered_tensor_number_methods());
  static auto tensor_methods = InitTensorPyMethodDefs();
  PyTensor_Type->tp_methods = tensor_methods.data();
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyTensor_Type) < 0) 
    << "PyTensor_Type not ready";
  Py_INCREF(PyTensor_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Tensor", reinterpret_cast<PyObject*>(PyTensor_Type)))
    << "Failed to add PyTensor_Type";
  
  static auto tensor_class_methods = InitTensorPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), tensor_class_methods.data()))
    << "Failed to add Tensor class methods";
}

} // namespace autograd
} // namespace hetu

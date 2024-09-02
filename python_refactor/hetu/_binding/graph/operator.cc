#include "hetu/_binding/graph/operator.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PyOperator_New(const Operator& op, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined && !op.is_defined()) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PyOperator_Type->tp_alloc(PyOperator_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyOperator";
    auto* self = reinterpret_cast<PyOperator*>(unsafe_self);
    new(&self->op) Operator();
    self->op = op;
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

PyObject* PyOperatorList_New(const OpList& ops, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  PyObject* ret = PyList_New(ops.size());
  HT_RUNTIME_ERROR_IF(!ret) << "Failed to alloc list";
  for (size_t i = 0; i < ops.size(); i++) {
    auto* op_obj = PyOperator_New(ops[i], return_none_if_undefined);
    PyList_SET_ITEM(ret, i, op_obj);
  }
  return ret;
  HT_PY_FUNC_END
}

void PyOperator_dealloc(PyOperator* self) {
  (&self->op)->~Operator();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyOperator_str(PyOperator* self) {
  HT_PY_FUNC_BEGIN
  // Compute?
  return PyUnicode_FromString(std::to_string(self->op));
  HT_PY_FUNC_END
}

PyObject* PyOperator_repr(PyOperator* self) {
  return PyOperator_str(self);
}

PyObject* PyOperator_id(PyOperator* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->op->id());
  HT_PY_FUNC_END
}

PyObject* PyOperator_name(PyOperator* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->op->name());
  HT_PY_FUNC_END
}

PyObject* PyOperator_type(PyOperator* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->op->type());
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyOperator_properties[] = {
  {PY_GET_SET_DEF_NAME("id"), (getter) PyOperator_id, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("name"), (getter) PyOperator_name, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("type"), (getter) PyOperator_type, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyOperator_methods[] = {
  // {"__reduce__", (PyCFunction) PyDevice_reduce, METH_NOARGS, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyOperator_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Operator", /* tp_name */
  sizeof(PyOperator), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyOperator_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyOperator_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyOperator_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyOperator_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyOperator_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyOperator_Type = &PyOperator_Type_obj;

void AddPyOperatorTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyOperator_Type) < 0) 
    << "PyOperator_Type not ready";
  Py_INCREF(PyOperator_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Operator", reinterpret_cast<PyObject*>(PyOperator_Type)))
    << "Failed to add PyOperator_Type";
}

PyObject* PyPushOpCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_op_ctx(Device eager_device=None, DeviceGroupHierarchy device_group_hierarchy=None, int stream_index=None, DataType dtype=None, List[Tensor] extra_deps=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    if (parsed_args.has(0))
      get_eager_device_ctx().push(parsed_args.get_device(0));
    if (parsed_args.has(1))
      get_dg_hierarchy_ctx().push(parsed_args.get_dg_hierarchy(1));
    if (parsed_args.has(2))
      get_stream_index_ctx().push(parsed_args.get_stream_index(2));
    if (parsed_args.has(3))
      get_dtype_ctx().push(parsed_args.get_dtype(3));
    if (parsed_args.has(4))
      get_extra_deps_ctx().push(parsed_args.get_tensor_list(4));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPopOpCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "pop_op_ctx(bool pop_eager_device=False, bool pop_device_group_hierarchy=False, bool pop_stream_index=False, bool pop_dtype=False, bool pop_extra_deps=False)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    if (parsed_args.get_bool_or_default(0))
      get_eager_device_ctx().pop();
    if (parsed_args.get_bool_or_default(1))
      get_dg_hierarchy_ctx().pop();
    if (parsed_args.get_bool_or_default(2))
      get_stream_index_ctx().pop();
    if (parsed_args.get_bool_or_default(3))
      get_dtype_ctx().pop();
    if (parsed_args.get_bool_or_default(4))
      get_extra_deps_ctx().pop();
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyOpCtx_methods[] = {
  {"push_op_ctx", (PyCFunction) PyPushOpCtx, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"pop_op_ctx", (PyCFunction) PyPopOpCtx, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {nullptr}
};

void AddOpContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyOpCtx_methods))
    << "Failed to add operator context managing methods";
}

ContextManager<TensorList>& get_extra_deps_ctx() {
  static ContextManager<TensorList> extra_deps_ctx;
  return extra_deps_ctx;
}

} // namespace graph
} // namespace hetu

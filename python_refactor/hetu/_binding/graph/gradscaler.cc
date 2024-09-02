#include "hetu/_binding/graph/gradscaler.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PyGradScaler_New(GradScaler scaler) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyGradScaler_Type->tp_alloc(PyGradScaler_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyGradScaler";
  auto* self = reinterpret_cast<PyGradScaler*>(unsafe_self);
  // new(&self->graph_id) GradScalerId();
  self->gradscaler = scaler;
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyGradScaler_dealloc(PyGradScaler* self) {
  (&self->gradscaler)->~GradScaler();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyGradScaler_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyGradScaler_Type->tp_alloc(PyGradScaler_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyGradScaler";
  auto* self = reinterpret_cast<PyGradScaler*>(unsafe_self);

  static PyArgParser parser({
    "GradScaler(double init_scale=65536.0, double growth_factor=2.0, double backoff_factor=0.5, int64_t growth_interval=2000, bool enabled=True)", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  if (parsed_args.signature_index() == 0) {
    new(&self->gradscaler) GradScaler();
    self->gradscaler = GradScaler(parsed_args.get_float64_or_default(0),
                                  parsed_args.get_float64_or_default(1),
                                  parsed_args.get_float64_or_default(2),
                                  parsed_args.get_int64_or_default(3), 
                                  parsed_args.get_bool_or_default(4));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyGradScaler_str(PyGradScaler* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString("gradscaler");
  HT_PY_FUNC_END
}

PyObject* PyGradScaler_repr(PyGradScaler* self) {
  return PyGradScaler_str(self);
}

// PyObject* PyGradScaler_id(PyGradScaler* self) {
//   HT_PY_FUNC_BEGIN
//   return PyLong_FromInteger(self->gradscaler_id);
//   HT_PY_FUNC_END
// }

PyObject* PyGradScaler_scale(PyGradScaler* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "scale(Tensor output)",
    "scale(TensorList outputs)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensor_New(self->gradscaler.scale(parsed_args.get_tensor_optional(0)));
    Py_RETURN_NONE;
  } else if (parsed_args.signature_index() == 1) {
    return PyTensorList_New(self->gradscaler.scale(parsed_args.get_tensor_list_or_empty(0)));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyGradScaler_minimize(PyGradScaler* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "minimize(SGDOptimizer op, Tensor loss, TensorList var_list=None, Tensor grad_loss=None)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensor_New(self->gradscaler.minimize(parsed_args.get_sgdoptimizer(0),
                        parsed_args.get_tensor_optional(1),
                        parsed_args.get_tensor_list_or_empty(2),
                        parsed_args.get_tensor_optional(3)));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyGradScaler_update(PyGradScaler* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "update(float new_scale=0)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensor_New(self->gradscaler.update(parsed_args.get_float64_or_default(0)));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyGradScaler_properties[] = {
  // {PY_GET_SET_DEF_NAME("id"), (getter) PyGradScaler_id, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyGradScaler_methods[] = {
  {"scale", (PyCFunction) PyGradScaler_scale, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"minimize", (PyCFunction) PyGradScaler_minimize, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"update", (PyCFunction) PyGradScaler_update, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyGradScaler_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.GradScaler", /* tp_name */
  sizeof(PyGradScaler), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyGradScaler_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyGradScaler_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyGradScaler_str, /* tp_str */
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
  PyGradScaler_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyGradScaler_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyGradScaler_pynew, /* tp_new */
};
PyTypeObject* PyGradScaler_Type = &PyGradScaler_Type_obj;

void AddPyGradScalerTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyGradScaler_Type) < 0) 
    << "PyGradScaler_Type not ready";
  Py_INCREF(PyGradScaler_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "GradScaler", reinterpret_cast<PyObject*>(PyGradScaler_Type)))
    << "Failed to add PyGradScaler_Type";
}

} // namespace graph
} // namespace hetu

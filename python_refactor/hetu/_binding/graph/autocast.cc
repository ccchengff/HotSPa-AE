#include "hetu/_binding/graph/autocast.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace graph {

PyObject* PyAutoCast_New(AutoCastId autocast_id) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyAutoCast_Type->tp_alloc(PyAutoCast_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyAutoCast";
  auto* self = reinterpret_cast<PyAutoCast*>(unsafe_self);
  // new(&self->graph_id) AutoCastId();
  self->autocast_id = autocast_id;
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyAutoCast_dealloc(PyDevice* self) {
  // (&self->graph_id)->~AutoCastId();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyAutoCast_str(PyAutoCast* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString("autocast");
  HT_PY_FUNC_END
}

PyObject* PyAutoCast_repr(PyAutoCast* self) {
  return PyAutoCast_str(self);
}

PyObject* PyAutoCast_id(PyAutoCast* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromInteger(self->autocast_id);
  HT_PY_FUNC_END
}

PyObject* PyAutoCast_get_autocast(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_autocast(int autocast_id)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    // Call `AutoCast::GetAutoCast` to check whether `graph_id` is valid
    return PyAutoCast_New(AutoCast::GetAutoCast(parsed_args.get_int64(0)).id());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyAutoCast_get_default_autocast(PyObject*) {
  HT_PY_FUNC_BEGIN
  return PyAutoCast_New(AutoCast::get_default_autocast().id());
  HT_PY_FUNC_END
}

PyObject* PyAutoCast_make_new_autocast(PyObject*, PyObject* args,
                                   PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "make_new_autocast(bool enable=True, DataType cast_type=None)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyAutoCast_New(
      AutoCast::MakeAutoCast(parsed_args.get_bool_or_default(0), parsed_args.get_dtype_or_else(1, DataType::UNDETERMINED))->id());
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPushAutoCastCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_autocast_ctx(int autocast_id)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    AutoCast::push_autocast_ctx(parsed_args.get_int64(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyPopAutoCastCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "pop_autocast_ctx()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    AutoCast::pop_autocast_ctx();
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyAutoCastCtx_methods[] = {
  {"get_autocast", (PyCFunction) PyAutoCast_get_autocast, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"get_default_autocast", (PyCFunction) PyAutoCast_get_default_autocast, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"make_new_autocast", (PyCFunction) PyAutoCast_make_new_autocast, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {"push_autocast_ctx", (PyCFunction) PyPushAutoCastCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pop_autocast_ctx", (PyCFunction) PyPopAutoCastCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE
PyGetSetDef PyAutoCast_properties[] = {
  {PY_GET_SET_DEF_NAME("id"), (getter) PyAutoCast_id, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyAutoCast_methods[] = {
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyAutoCast_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.AutoCast", /* tp_name */
  sizeof(PyAutoCast), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyAutoCast_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyAutoCast_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyAutoCast_str, /* tp_str */
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
  PyAutoCast_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyAutoCast_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyAutoCast_Type = &PyAutoCast_Type_obj;

void AddPyAutoCastTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyAutoCast_Type) < 0) 
    << "PyAutoCast_Type not ready";
  Py_INCREF(PyAutoCast_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "AutoCast", reinterpret_cast<PyObject*>(PyAutoCast_Type)))
    << "Failed to add PyAutoCast_Type";
}

void AddAutoCastContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyAutoCastCtx_methods))
    << "Failed to add graph context managing methods";
}

} // namespace graph
} // namespace hetu

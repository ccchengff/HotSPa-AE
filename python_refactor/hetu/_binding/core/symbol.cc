#include "hetu/_binding/core/symbol.h"
#include "hetu/_binding/core/device.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {

PyObject* PyIntSymbol_New(const IntSymbol& int_symbol) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyIntSymbol_Type->tp_alloc(PyIntSymbol_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyIntSymbol";
  auto* self = reinterpret_cast<PyIntSymbol*>(unsafe_self);
  new (&self->int_symbol) IntSymbol(int_symbol);
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END  
}

PyObject* PySyShape_New(const SyShape& symbol_shape) {
  HT_PY_FUNC_BEGIN
  auto len = symbol_shape.size();
  auto* ret = PyList_New(len);
  for (size_t i = 0; i < len; i++) {
    auto* int_symbol = PyIntSymbol_New(symbol_shape[i]);
    PyList_SET_ITEM(ret, i, int_symbol);
  }
  return ret;
  HT_PY_FUNC_END  
}

PyObject* PyIntSymbol_pynew(PyTypeObject* type, PyObject* args, 
                                    PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyIntSymbol_Type->tp_alloc(PyIntSymbol_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyIntSymbol";  
  auto* self = reinterpret_cast<PyIntSymbol*>(unsafe_self);
  static PyArgParser parser({
    "IntSymbol(int val)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    int32_t val = parsed_args.get_int64(0);
    new (&self->int_symbol) IntSymbol(val);
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyIntSymbol_dealloc(PyIntSymbol* self) {
  (&self->int_symbol)->~IntSymbol();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyIntSymbol_str(PyIntSymbol* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString(self->int_symbol.symbol_info());
  HT_PY_FUNC_END
}

PyObject* PyIntSymbol_repr(PyIntSymbol* self) {
  return PyIntSymbol_str(self);
}

PyObject* PyIntSymbol_is_leaf(PyIntSymbol* self) {
  HT_PY_FUNC_BEGIN
  Py_RETURN_BOOLEAN_COND(self->int_symbol->is_leaf());
  HT_PY_FUNC_END
}

PyObject* PyIntSymbol_get_data(PyIntSymbol* self) {
  HT_PY_FUNC_BEGIN
  if (!self->int_symbol.is_defined()) {
    HT_LOG_WARN << "You are getting the data from a nullptr symbol, "
      << "please attach that symbol to some other symbol or a specific data in advance.";
    Py_RETURN_NONE;
  }
  return PyLong_FromInteger(self->int_symbol->get_val());
  HT_PY_FUNC_END
}

PyObject* PyIntSymbol_set_data(PyIntSymbol* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "set_data(int data)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto data = parsed_args.get_int64(0);
    if (self->int_symbol.is_defined() && !self->int_symbol->is_leaf()) {
      HT_RUNTIME_ERROR << "You can't set the data of a non-leaf symbol. " 
        << "You may use reset_data method to attach the symbol to a new data "
        << "and turn it into a leaf.";
    }
    self->int_symbol = data;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyIntSymbol_reset_data(PyIntSymbol* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "reset_data(int data)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    auto data = parsed_args.get_int64(0);
    self->int_symbol.reset();
    self->int_symbol = data;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyIntSymbol_properties[] = {
  {PY_GET_SET_DEF_NAME("is_leaf"), (getter) PyIntSymbol_is_leaf, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("data"), (getter) PyIntSymbol_get_data, nullptr, nullptr, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyIntSymbol_methods[] = {
  {"get_data", (PyCFunction) PyIntSymbol_get_data, METH_NOARGS, nullptr },
  {"set_data", (PyCFunction) PyIntSymbol_set_data, METH_VARARGS | METH_KEYWORDS, nullptr },  
  {"reset_data", (PyCFunction) PyIntSymbol_reset_data, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyIntSymbol_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.IntSymbol", /* tp_name */
  sizeof(PyIntSymbol), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyIntSymbol_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyIntSymbol_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyIntSymbol_str, /* tp_str */
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
  PyIntSymbol_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyIntSymbol_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyIntSymbol_pynew, /* tp_new */
};
PyTypeObject* PyIntSymbol_Type = &PyIntSymbol_Type_obj;

/******************************************************
 * Arithmetics
 ******************************************************/

PyObject* PyIntSymbol_add(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "add(IntSymbol x, IntSymbol y)",
    "add(IntSymbol x, int y)",
    "add(int x, IntSymbol y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol +
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else if (parsed_args.signature_index() == 1) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol +
      Int64_FromPyLong(y)
    );
  } else if (parsed_args.signature_index() == 2) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol +
      Int64_FromPyLong(x)
    );
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_INT_SYMBOL_NUMBER_METHOD(nb_add, (binaryfunc) PyIntSymbol_add);

PyObject* PyIntSymbol_sub(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "sub(IntSymbol x, IntSymbol y)",
    "sub(IntSymbol x, int y)",
    "sub(int x, IntSymbol y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol -
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else if (parsed_args.signature_index() == 1) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol -
      Int64_FromPyLong(y)
    );
  } else if (parsed_args.signature_index() == 2) {
    return PyIntSymbol_New(
      IntSymbol(Int64_FromPyLong(x)) -
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_INT_SYMBOL_NUMBER_METHOD(nb_subtract, (binaryfunc) PyIntSymbol_sub);

PyObject* PyIntSymbol_neg(PyObject* x) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"neg(IntSymbol x)"});
  auto parsed_args = parser.parse(x, nullptr, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyIntSymbol_New(
      IntSymbol(0) -
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol
    );
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_INT_SYMBOL_NUMBER_METHOD(nb_negative, (unaryfunc) PyIntSymbol_neg);

PyObject* PyIntSymbol_mul(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "mul(IntSymbol x, IntSymbol y)",
    "mul(IntSymbol x, int y)",
    "mul(int x, IntSymbol y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol *
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else if (parsed_args.signature_index() == 1) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol *
      Int64_FromPyLong(y)
    );
  } else if (parsed_args.signature_index() == 2) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol *
      Int64_FromPyLong(x)
    );
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_INT_SYMBOL_NUMBER_METHOD(nb_multiply, (binaryfunc) PyIntSymbol_mul);

PyObject* PyIntSymbol_div(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "div(IntSymbol x, IntSymbol y)",
    "div(IntSymbol x, int y)",
    "div(int x, IntSymbol y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol /
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else if (parsed_args.signature_index() == 1) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol /
      Int64_FromPyLong(y)
    );
  } else if (parsed_args.signature_index() == 2) {
    return PyIntSymbol_New(
      IntSymbol(Int64_FromPyLong(x)) /
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_INT_SYMBOL_NUMBER_METHOD(nb_true_divide, (binaryfunc) PyIntSymbol_div);

PyObject* PyIntSymbol_rem(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "rem(IntSymbol x, IntSymbol y)",
    "rem(IntSymbol x, int y)",
    "rem(int x, IntSymbol y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol %
      reinterpret_cast<PyIntSymbol*>(y)->int_symbol
    );
  } else if (parsed_args.signature_index() == 1) {
    return PyIntSymbol_New(
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol %
      Int64_FromPyLong(y)
    );
  } else if (parsed_args.signature_index() == 2) {
    return PyIntSymbol_New(
      IntSymbol(Int64_FromPyLong(y)) %
      reinterpret_cast<PyIntSymbol*>(x)->int_symbol
    );
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_INT_SYMBOL_NUMBER_METHOD(nb_remainder, (binaryfunc) PyIntSymbol_rem);

/******************************************************
 * Wrap It Up
 ******************************************************/

void AddPyIntSymbolTypeToModule(py::module_& module) {
  PyIntSymbol_Type->tp_as_number = &(get_registered_int_symbol_number_methods());
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyIntSymbol_Type) < 0) 
    << "PyIntSymbol_Type not ready";
  Py_INCREF(PyIntSymbol_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "IntSymbol", 
      reinterpret_cast<PyObject*>(PyIntSymbol_Type)))
    << "Failed to add PyIntSymbol_Type";
}

} // namespace hetu

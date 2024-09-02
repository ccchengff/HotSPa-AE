#pragma once

#include <Python.h>
#include "hetu/core/dtype.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/context_manager.h"

namespace hetu {

struct PyDataType {
  PyObject_HEAD;
  DataType dtype;
};

extern PyTypeObject* PyDataType_Type;

inline bool PyDataType_Check(PyObject* obj) {
  return PyDataType_Type && PyObject_TypeCheck(obj, PyDataType_Type);
}

inline bool PyDataType_CheckExact(PyObject* obj) {
  return PyDataType_Type && obj->ob_type == PyDataType_Type;
}

inline DataType DataType_FromPyDataType(PyObject* obj) {
  return reinterpret_cast<PyDataType*>(obj)->dtype;
}

PyObject* PyDataType_New(DataType dtype);

void AddPyDataTypeTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyDataType(PyObject* obj) {
  return PyDataType_Check(obj) || \
    obj == reinterpret_cast<PyObject*>(&PyFloat_Type) || \
    obj == reinterpret_cast<PyObject*>(&PyLong_Type) || \
    obj == reinterpret_cast<PyObject*>(&PyBool_Type);
}

inline DataType DataType_FromPyObject(PyObject* obj) {
  if (PyDataType_Check(obj)) {
    return reinterpret_cast<PyDataType*>(obj)->dtype;
  } else if (obj == reinterpret_cast<PyObject*>(&PyFloat_Type)) {
    return kDouble;
  } else if (obj == reinterpret_cast<PyObject*>(&PyLong_Type)) {
    return kLong;
  } else if (obj == reinterpret_cast<PyObject*>(&PyBool_Type)) {
    return kBool;
  } else {
    HT_VALUE_ERROR << "Cannot cast " << Py_TYPE(obj)->tp_name << " as dtype";
    __builtin_unreachable();
  }
}

/******************************************************
 * For contextlib usage
 ******************************************************/

ContextManager<DataType>& get_dtype_ctx();

} // namespace hetu

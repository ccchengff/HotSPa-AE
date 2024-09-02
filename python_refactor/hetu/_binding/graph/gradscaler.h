#pragma once

#include <Python.h>
#include "hetu/graph/autocast/gradscaler.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PyGradScaler {
  PyObject_HEAD;
  GradScaler gradscaler;
};

extern PyTypeObject* PyGradScaler_Type;

inline bool PyGradScaler_Check(PyObject* obj) {
  return PyGradScaler_Type && PyObject_TypeCheck(obj, PyGradScaler_Type);
}

inline bool PyGradScaler_CheckExact(PyObject* obj) {
  return PyGradScaler_Type && obj->ob_type == PyGradScaler_Type;
}

PyObject* PyGradScaler_New(GradScaler scaler);

void AddPyGradScalerTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyGradScaler(PyObject* obj) {
  return PyGradScaler_Check(obj);
}

inline GradScaler GradScaler_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyGradScaler*>(obj)->gradscaler;
}

} // namespace graph
} // namespace hetu

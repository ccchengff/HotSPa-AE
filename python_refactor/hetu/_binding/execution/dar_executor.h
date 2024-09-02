#pragma once

#include <Python.h>
#include "hetu/execution/dar_executor.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/autograd/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace execution {

struct PyDARExecutor {
  PyObject_HEAD;
  std::shared_ptr<DARExecutor> executor;
};

extern PyTypeObject* PyDARExecutor_Type;

inline bool PyDARExecutor_Check(PyObject* obj) {
  return PyDARExecutor_Type && PyObject_TypeCheck(obj, PyDARExecutor_Type);
}

inline bool PyDARExecutor_CheckExact(PyObject* obj) {
  return PyDARExecutor_Type && obj->ob_type == PyDARExecutor_Type;
}

void AddPyDARExecutorTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyFeedDict(PyObject* obj) {
  if (PyDict_Check(obj)) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
      if (!CheckPyTensor(key))
        return false;
      if (!CheckPyNDArray(value) && !CheckNumpyArray(value))
        return false;
    }
    return true;
  }
  return false;
}

inline FeedDict FeedDict_FromPyObject(PyObject* obj) {
  FeedDict feed_dict;
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    TensorId k = Tensor_FromPyObject(key)->id();
    NDArray v = CheckPyNDArray(value) ? NDArray_FromPyObject(value)
                                      : NDArrayFromNumpy(value);
    feed_dict.insert({k, v});
  }
  return feed_dict;
}

} // namespace execution
} // namespace hetu

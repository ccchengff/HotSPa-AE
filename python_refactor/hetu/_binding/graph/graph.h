#pragma once

#include <Python.h>
#include "hetu/graph/graph.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/utils/numpy.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PyGraph {
  PyObject_HEAD;
  GraphId graph_id;
};

extern PyTypeObject* PyGraph_Type;

inline bool PyGraph_Check(PyObject* obj) {
  return PyGraph_Type && PyObject_TypeCheck(obj, PyGraph_Type);
}

inline bool PyGraph_CheckExact(PyObject* obj) {
  return PyGraph_Type && obj->ob_type == PyGraph_Type;
}

PyObject* PyGraph_New(GraphId graph_id);

void AddPyGraphTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyGraph(PyObject* obj) {
  return PyGraph_Check(obj);
}

inline GraphId GraphId_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyGraph*>(obj)->graph_id;
}

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
  int64_t cnt = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": processing element " << cnt++ << " in FeedDict...";
    TensorId k = Tensor_FromPyObject(key)->id();
    NDArray v = CheckPyNDArray(value) ? NDArray_FromPyObject(value)
                                      : NDArrayFromNumpy(value);
    feed_dict.insert({k, v});
  }
  return feed_dict;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddGraphContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hetu

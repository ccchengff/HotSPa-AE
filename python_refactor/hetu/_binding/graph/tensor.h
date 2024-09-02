#pragma once

#include <Python.h>
#include "hetu/graph/operator.h"
#include "hetu/graph/tensor.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PyTensor {
  PyObject_HEAD;
  Tensor tensor;
};

extern PyTypeObject* PyTensor_Type;

inline bool PyTensor_Check(PyObject* obj) {
  return PyTensor_Type && PyObject_TypeCheck(obj, PyTensor_Type);
}

inline bool PyTensor_CheckExact(PyObject* obj) {
  return PyTensor_Type && obj->ob_type == PyTensor_Type;
}

PyObject* PyTensor_New(Tensor&& tensor, bool return_none_if_undefined = true);

PyObject* PyTensorList_New(TensorList&& tensors,
                           bool return_none_if_undefined = true);

void AddPyTensorTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyTensor(PyObject* obj) {
  return PyTensor_Check(obj);
}

inline Tensor Tensor_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyTensor*>(obj)->tensor;
}

inline bool CheckPyTensorList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyTensor(item))
        return false;
    }
    return true;
  }
  return false;
}

inline TensorList TensorList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  TensorList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Tensor_FromPyObject(item);
  }
  return ret;
}

inline PyObject* PyObject_FromOperatorOutputs(Tensor&& tensor) {
  return PyTensor_New(std::move(tensor));
}

inline PyObject* PyObject_FromOperatorOutputs(TensorList&& tensors) {
  PyObject* ret = PyTuple_New(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++)
    PyTuple_SET_ITEM(ret, i, PyTensor_New(std::move(tensors[i])));
  return ret;
}

} // namespace graph
} // namespace hetu

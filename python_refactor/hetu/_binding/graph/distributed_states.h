#pragma once

#include <Python.h>
#include "hetu/graph/distributed_states.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {
namespace graph {

struct PyDistributedStates {
  PyObject_HEAD;
  DistributedStates distributed_states;
};

struct PyDistributedStatesUnion {
  PyObject_HEAD;
  DistributedStatesUnion ds_union;
};

extern PyTypeObject* PyDistributedStates_Type;

extern PyTypeObject* PyDistributedStatesUnion_Type;

inline bool PyDistributedStates_Check(PyObject* obj) {
  return PyDistributedStates_Type && PyObject_TypeCheck(obj, PyDistributedStates_Type);
}

inline bool PyDistributedStatesUnion_Check(PyObject* obj) {
  return PyDistributedStatesUnion_Type && PyObject_TypeCheck(obj, PyDistributedStatesUnion_Type);
}

inline bool PyDistributedStates_CheckExact(PyObject* obj) {
  return PyDistributedStates_Type && obj->ob_type == PyDistributedStates_Type;
}

inline bool PyDistributedStatesUnion_CheckExact(PyObject* obj) {
  return PyDistributedStatesUnion_Type && obj->ob_type == PyDistributedStatesUnion_Type;
}

PyObject* PyDistributedStates_New(const DistributedStates& ds);

PyObject* PyDistributedStatesList_New(const DistributedStatesList& ds_list);

PyObject* PyDistributedStatesUnion_New(const DistributedStatesUnion& ds_union);

PyObject* PyDistributedStatesHierarchy_New(const DistributedStatesHierarchy& ds_hierarchy);

void AddPyDistributedStatesTypeToModule(py::module_& module);

void AddPyDistributedStatesUnionTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyDistributedStates(PyObject* obj) {
  return PyDistributedStates_Check(obj);
}

inline bool CheckPyDistributedStatesUnion(PyObject* obj) {
  return PyDistributedStatesUnion_Check(obj);
}

inline DistributedStates DistributedStates_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyDistributedStates*>(obj)->distributed_states;
}

inline DistributedStatesUnion DistributedStatesUnion_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyDistributedStatesUnion*>(obj)->ds_union;
}

inline bool CheckPyDistributedStatesList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyDistributedStates(item))
        return false;
    }
    return true;
  }
  return false;
}

inline DistributedStatesList DistributedStatesList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  DistributedStatesList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = DistributedStates_FromPyObject(item);
  }
  return ret;
}

inline bool CheckPyDistributedStatesHierarchy(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyDistributedStatesUnion(item))
        return false;
    }
    return true;
  }
  return false;
}

inline DistributedStatesHierarchy DistributedStatesHierarchy_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  DistributedStatesHierarchy ret;
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    // ret.add(DistributedStatesUnion(DistributedStatesList_FromPyObject(item)));
    ret.add(DistributedStatesUnion_FromPyObject(item));
  }
  return ret;
}

} // namespace graph
} // namespace hetu

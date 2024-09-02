#pragma once

#include <Python.h>
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/except.h"

namespace hetu {

#if PY_VERSION_HEX < 0x03070000
#define PY_GET_SET_DEF_NAME(name) const_cast<char*>(name)
#else
#define PY_GET_SET_DEF_NAME(name) (name)
#endif

#define Py_RETURN_BOOLEAN_COND(cond)                                          \
  do { if (cond) { Py_RETURN_TRUE; } else { Py_RETURN_FALSE; } } while (0)

#define Py_RETURN_UNICODE_FROM_OSS(x)                                         \
  do {                                                                        \
    std::ostringstream oss; oss << x;                                         \
    return PyUnicode_FromString(oss.str());                                   \
  } while (0)

#define Py_RETURN_RICH_CMP_EQ_ONLY(x, y, op, type_x, type_y)                  \
  do {                                                                        \
    switch (op) {                                                             \
      case Py_EQ:                                                             \
        Py_RETURN_BOOLEAN_COND((x) == (y));                                   \
      case Py_NE:                                                             \
        Py_RETURN_BOOLEAN_COND(!((x) == (y)));                                \
      case Py_LT:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for LT between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      case Py_LE:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for LE between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      case Py_GT:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for GT between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      case Py_GE:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for GE between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      default:                                                                \
        HT_VALUE_ERROR << "Unknown comparison op: " << (op);                  \
        Py_RETURN_NONE;                                                       \
    }                                                                         \
  } while (0)

#define Py_RETURN_RICH_CMP_LT_ONLY(x, y, op, type_x, type_y)                  \
  do {                                                                        \
    switch (op) {                                                             \
      case Py_LT:                                                             \
        Py_RETURN_BOOLEAN_COND((x) < (y));                                    \
      case Py_GE:                                                             \
        Py_RETURN_BOOLEAN_COND(!((x) < (y)));                                 \
      case Py_LE:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for LE between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      case Py_EQ:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for EQ between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      case Py_NE:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for NE between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      case Py_GT:                                                             \
        HT_NOT_IMPLEMENTED << "Comparison for GT between " << (type_x)        \
          << " and " << (type_y) << " is not supported";                      \
      default:                                                                \
        HT_VALUE_ERROR << "Unknown comparison op: " << (op);                  \
        Py_RETURN_NONE;                                                       \
    }                                                                         \
  } while (0)

#define Py_RETURN_RICH_CMP_EQ_AND_LT_ONLY(x, y, op)                           \
  do {                                                                        \
    switch (op) {                                                             \
      case Py_LT:                                                             \
        Py_RETURN_BOOLEAN_COND((x) < (y));                                    \
      case Py_LE:                                                             \
        Py_RETURN_BOOLEAN_COND((x) < (y) || (x) == (y));                      \
      case Py_EQ:                                                             \
        Py_RETURN_BOOLEAN_COND((x) == (y));                                   \
      case Py_NE:                                                             \
        Py_RETURN_BOOLEAN_COND(!((x) == (y)));                                \
      case Py_GT:                                                             \
        Py_RETURN_BOOLEAN_COND(!((x) < (y) || (x) == (y)));                   \
      case Py_GE:                                                             \
        Py_RETURN_BOOLEAN_COND(!((x) < (y)));                                 \
      default:                                                                \
        HT_VALUE_ERROR << "Unknown comparison op: " << (op);                  \
        Py_RETURN_NONE;                                                       \
    }                                                                         \
  } while (0)

#define Py_RETURN_RICH_CMP_COMPLETE(x, y, op)                                 \
  do {                                                                        \
    switch (op) {                                                             \
      case Py_LT:                                                             \
        Py_RETURN_BOOLEAN_COND((x) < (y));                                    \
      case Py_LE:                                                             \
        Py_RETURN_BOOLEAN_COND((x) <= (y));                                   \
      case Py_EQ:                                                             \
        Py_RETURN_BOOLEAN_COND((x) == (y));                                   \
      case Py_NE:                                                             \
        Py_RETURN_BOOLEAN_COND((x) != (y));                                   \
      case Py_GT:                                                             \
        Py_RETURN_BOOLEAN_COND((x) > (y));                                    \
      case Py_GE:                                                             \
        Py_RETURN_BOOLEAN_COND((x) >= (y));                                   \
      default:                                                                \
        HT_VALUE_ERROR << "Unknown comparison op: " << (op);                  \
        Py_RETURN_NONE;                                                       \
    }                                                                         \
  } while (0)

inline void AddPyMethodDef(std::vector<PyMethodDef>& vec, PyMethodDef method) {
  HT_ASSERT(!vec.empty() && vec.back().ml_name == nullptr) 
    << "Vector of PyMethodDef not ended with Sentinel";
  vec.pop_back();
  vec.push_back(method);
  vec.push_back({nullptr});
}

inline void AddPyMethodDefs(std::vector<PyMethodDef>& vec, 
                            const PyMethodDef* methods) {
  HT_ASSERT(!vec.empty() && vec.back().ml_name == nullptr) 
    << "Vector of PyMethodDef not ended with Sentinel";
  vec.pop_back();
  while (methods->ml_name != nullptr) {
    vec.push_back(*methods);
    methods++;
  }
  vec.push_back({nullptr});
}

inline void AddPyMethodDefs(std::vector<PyMethodDef>& vec, 
                            const std::vector<PyMethodDef> methods) {
  HT_ASSERT(!vec.empty() && vec.back().ml_name == nullptr) 
    << "Vector of PyMethodDef not ended with Sentinel";
  HT_ASSERT(!methods.empty() && methods.back().ml_name == nullptr) 
    << "Vector of PyMethodDef not ended with Sentinel";
  AddPyMethodDefs(vec, methods.data());
}

} // namespace hetu

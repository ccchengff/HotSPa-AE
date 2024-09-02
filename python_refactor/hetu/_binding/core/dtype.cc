#include "hetu/_binding/core/dtype.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {

PyObject* PyDataType_New(DataType dtype) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDataType_Type->tp_alloc(PyDataType_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDataType";  
  auto* self = reinterpret_cast<PyDataType*>(unsafe_self);
  self->dtype = dtype;
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

PyObject* PyDataType_str(PyDataType* self) {
  HT_PY_FUNC_BEGIN
  return PyUnicode_FromString("hetu." + DataType2Str(self->dtype));
  HT_PY_FUNC_END
}

PyObject* PyDataType_repr(PyDataType* self) {
  return PyDataType_str(self);
}

PyObject* PyDataType_rich_cmp(PyObject* obj_x, PyObject* obj_y, int op) {
  HT_PY_FUNC_BEGIN
  if (!PyDataType_Check(obj_x) || !PyDataType_Check(obj_y)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto& x = reinterpret_cast<PyDataType*>(obj_x)->dtype;
  auto& y = reinterpret_cast<PyDataType*>(obj_y)->dtype;
  Py_RETURN_RICH_CMP_COMPLETE(x, y, op);
  HT_PY_FUNC_END
}

Py_ssize_t PyDataType_hash(PyDataType* self) {
  HT_PY_FUNC_BEGIN
  return static_cast<Py_ssize_t>(std::hash<DataType>()(self->dtype));
  HT_PY_FUNC_RETURN(-1)
}

PyObject* PyDataType_reduce(PyDataType* self) {
  HT_PY_FUNC_BEGIN
  return PyDataType_str(self);
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyDataType_methods[] = {
  {"__reduce__", (PyCFunction) PyDataType_reduce, METH_NOARGS, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyDataType_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.dtype", /* tp_name */
  sizeof(PyDataType), /* tp_basicsize */
  0, /* tp_itemsize */
  nullptr, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyDataType_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  (hashfunc) PyDataType_hash, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyDataType_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  (richcmpfunc) PyDataType_rich_cmp, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  PyDataType_methods, /* tp_methods */
  nullptr, /* tp_members */
  nullptr, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyDataType_Type = &PyDataType_Type_obj;

void AddPyDataTypeTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyDataType_Type) < 0) 
    << "PyDataType_Type not ready";
  Py_INCREF(PyDataType_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "dtype", reinterpret_cast<PyObject*>(PyDataType_Type)))
    << "Failed to add PyDataType_Type";
  for (int i = 0; i < static_cast<int>(NUM_DATA_TYPES); i++) {
    DataType dtype = static_cast<DataType>(i);
    auto dtype_obj = PyDataType_New(dtype);
    HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
        module.ptr(), DataType2Str(dtype).c_str(), dtype_obj))
      << "Failed to add " << dtype;
  }
}

ContextManager<DataType>& get_dtype_ctx() {
  static ContextManager<DataType> dtype_ctxs;
  return dtype_ctxs;
}

} // namespace hetu

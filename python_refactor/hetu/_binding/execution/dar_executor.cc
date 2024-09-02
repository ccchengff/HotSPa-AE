#include "hetu/_binding/execution/dar_executor.h"
#include "hetu/_binding/autograd/operator.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"

namespace hetu {
namespace execution {

PyObject* PyDARExecutor_pynew(PyTypeObject* type, PyObject* args,
                              PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDARExecutor_Type->tp_alloc(PyDARExecutor_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDARExecutor";  
  auto* self = reinterpret_cast<PyDARExecutor*>(unsafe_self);
  static PyArgParser parser({
    "Executor(Device local_device, DeviceGroup device_group=None, List[Tensor] losses=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    new (&self->executor) std::shared_ptr<DARExecutor>(new DARExecutor(
      parsed_args.get_device(0), 
      parsed_args.get_device_group_or_peek(1).value_or(DeviceGroup())
    ));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

void PyDARExecutor_dealloc(PyDARExecutor* self) {
  self->executor = nullptr;
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyDARExecutor_run(PyDARExecutor* self, PyObject* args,
                            PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "run(Tensor fetch, FeedDict feed_dict=None)", 
    "run(List[Tensor] fetches, FeedDict feed_dict=None)"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyNDArrayList_New(self->executor->Run(
      {parsed_args.get_tensor(0)}, 
      parsed_args.get_feed_dict_or_empty(1)));
  } else if (parsed_args.signature_index() == 1) {
    return PyNDArrayList_New(self->executor->Run(
      parsed_args.get_tensor_list(0), 
      parsed_args.get_feed_dict_or_empty(1)));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

PyObject* PyDARExecutor_local_device(PyDARExecutor* self) {
  HT_PY_FUNC_BEGIN
  return PyDevice_New(self->executor->local_device());
  HT_PY_FUNC_END
}

PyObject* PyDARExecutor_device_group(PyDARExecutor* self) {
  HT_PY_FUNC_BEGIN
  return PyDeviceGroup_New(self->executor->device_group());
  HT_PY_FUNC_END
}

PyObject* PyDARExecutor_local_topo_order(PyDARExecutor* self) {
  HT_PY_FUNC_BEGIN
  return PyOperatorList_New(self->executor->local_topo_order());
  HT_PY_FUNC_END
}

PyObject* PyDARExecutor_global_topo_order(PyDARExecutor* self) {
  HT_PY_FUNC_BEGIN
  return PyOperatorList_New(self->executor->global_topo_order());
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyGetSetDef PyDARExecutor_properties[] = {
  {PY_GET_SET_DEF_NAME("local_device"), (getter) PyDARExecutor_local_device, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("device_group"), (getter) PyDARExecutor_device_group, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("local_topo_order"), (getter) PyDARExecutor_local_topo_order, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("global_topo_order"), (getter) PyDARExecutor_global_topo_order, nullptr, nullptr, nullptr}, 
  {nullptr}
};

// NOLINTNEXTLINE
PyMethodDef PyDARExecutor_methods[] = {
  {"run", (PyCFunction) PyDARExecutor_run, METH_VARARGS | METH_KEYWORDS, nullptr }, 
  {nullptr}
};

// NOLINTNEXTLINE
PyTypeObject PyDARExecutor_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.DARExecutor", /* tp_name */
  sizeof(PyDARExecutor), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyDARExecutor_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  nullptr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  nullptr, /* tp_str */
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
  PyDARExecutor_methods, /* tp_methods */
  nullptr, /* tp_members */
  PyDARExecutor_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyDARExecutor_pynew, /* tp_new */
};
PyTypeObject* PyDARExecutor_Type = &PyDARExecutor_Type_obj;

void AddPyDARExecutorTypeToModule(py::module_& module) {
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyDARExecutor_Type) < 0) 
    << "PyDARExecutor_Type not ready";
  Py_INCREF(PyDARExecutor_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "DARExecutor", reinterpret_cast<PyObject*>(PyDARExecutor_Type)))
    << "Failed to add PyDARExecutor_Type";
}

} // namespace execution
} // namespace hetu

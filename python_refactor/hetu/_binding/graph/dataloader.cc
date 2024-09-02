#include "hetu/_binding/graph/tensor.h"
#include "hetu/_binding/graph/dataloader.h"
#include "hetu/_binding/graph/tensor_ctor.h"
#include "hetu/_binding/graph/graph.h"
#include "hetu/_binding/core/ndarray.h"
#include "hetu/_binding/constants.h"
#include "hetu/_binding/utils/function_registry.h"
#include "hetu/_binding/utils/pybind_common.h"
#include "hetu/_binding/utils/python_primitives.h"
#include "hetu/_binding/utils/except.h"
#include "hetu/_binding/utils/decl_utils.h"
#include "hetu/_binding/utils/arg_parser.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

PyObject* PyDataloader_New(Dataloader&& dataloader, bool return_none_if_undefined) {
  HT_PY_FUNC_BEGIN
  if (return_none_if_undefined) {
    Py_RETURN_NONE;
  } else {
    auto* unsafe_self = PyDataloader_Type->tp_alloc(PyDataloader_Type, 0);
    HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDataloader";
    auto* self = reinterpret_cast<PyDataloader*>(unsafe_self);
    new(&self->dataloader) Dataloader();
    self->dataloader = std::move(dataloader);
    return reinterpret_cast<PyObject*>(self);
  }
  HT_PY_FUNC_END
}

inline PyObject* PyDataloader_pynew(PyTypeObject* type, PyObject* args, 
                                      PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  auto* unsafe_self = PyDataloader_Type->tp_alloc(PyDataloader_Type, 0);
  HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDataloader";
  auto* self = reinterpret_cast<PyDataloader*>(unsafe_self);

  static PyArgParser parser({
    "Dataloader(numpy.array raw_data, int batch_size, int num_workers=0, std::string name='default', bool shuffle=false, bool drop_last=true)", 
    "Dataloader(NDArray raw_data, int batch_size, int num_workers=0, std::string name='default', bool shuffle=false, bool drop_last=true)", 
  });
  auto parsed_args = parser.parse(args, kwargs);
  
  // Question: During device placement, the ndarray data will be copied 
  // if it does not match the expected dtype or device. 
  // Can we defer the copy to device placement?
  if (parsed_args.signature_index() == 0) {
    new(&self->dataloader) Dataloader();
    auto* array_obj = parsed_args.get_numpy_array(0);
    self->dataloader = Dataloader(NDArrayFromNumpy(array_obj),
                                  parsed_args.get_int64(1),
                                  parsed_args.get_int64_or_default(2), 
                                  parsed_args.get_string_or_default(3), 
                                  parsed_args.get_bool_or_default(4), 
                                  parsed_args.get_bool_or_default(5));
  } else if (parsed_args.signature_index() == 1) {
    new(&self->dataloader) Dataloader();
    self->dataloader = Dataloader(parsed_args.get_ndarray(0),
                                  parsed_args.get_int64(1),
                                  parsed_args.get_int64_or_default(2), 
                                  parsed_args.get_string_or_default(3), 
                                  parsed_args.get_bool_or_default(4), 
                                  parsed_args.get_bool_or_default(5));
  } else {
    Py_TYPE(self)->tp_free(self);
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }

  return reinterpret_cast<PyObject*>(self);
  HT_PY_FUNC_END
}

// PyObject* PyDataloader_make_subclass(PyObject*, PyObject* args, PyObject* kwargs) {
//   HT_PY_FUNC_BEGIN
//   static PyArgParser parser({
//     "_make_subclass(PyObject* cls, Dataloader data, bool requires_grad=false)"
//   });
//   auto parsed_args = parser.parse(args, kwargs);
//   if (parsed_args.signature_index() == 0) {
//     PyObject* cls = parsed_args.get_py_obj(0);
//     HT_TYPE_ERROR_IF(!PyType_Check(cls))
//       << "Expected argument \"cls\" to be a type (got " 
//       << Py_TYPE(cls)->tp_name << ")";
//     PyTypeObject* cls_type = reinterpret_cast<PyTypeObject*>(cls);
//     HT_TYPE_ERROR_IF(!PyType_IsSubtype(cls_type, PyDataloader_Type))
//       << "Type " << cls_type->tp_name << " is not derived from hetu.Dataloader";

//     auto* unsafe_self = cls_type->tp_alloc(cls_type, 0);
//     HT_RUNTIME_ERROR_IF(!unsafe_self) << "Failed to alloc PyDataloader";
//     // Question: Is the casting safe?
//     auto* self = reinterpret_cast<PyDataloader*>(unsafe_self);
//     new(&self->tensor) Dataloader();

//     self->tensor = parsed_args.get_tensor(1);
//     HT_TYPE_ERROR_IF(!self->tensor->is_variable())
//       << "Subclass of hetu.Dataloader must be created from a variable. "
//       << "Please detach the tensor first.";
//     HT_NOT_IMPLEMENTED;
//     // self->tensor->set_trainable(parsed_args.get_bool_or_default(2));
    
//     return reinterpret_cast<PyObject*>(self);
//   } else {
//     HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
//     __builtin_unreachable();
//   }
//   HT_PY_FUNC_END
// }

void PyDataloader_dealloc(PyDataloader* self) {
  (&self->dataloader)->~Dataloader();
  Py_TYPE(self)->tp_free(self);
}

PyObject* PyDataloader_str(PyDataloader* self) {
  HT_PY_FUNC_BEGIN
  // Compute?
  return PyUnicode_FromString("dataloader");
  HT_PY_FUNC_END
}

PyObject* PyDataloader_repr(PyDataloader* self) {
  return PyDataloader_str(self);
}

PyObject* PyDataloader_num_workers(PyDataloader* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromLong(static_cast<long>(self->dataloader.num_workers()));
  HT_PY_FUNC_END
}

PyObject* PyDataloader_sample_num(PyDataloader* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromLong(static_cast<long>(self->dataloader.sample_num()));
  HT_PY_FUNC_END
}

PyObject* PyDataloader_batch_num(PyDataloader* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromLong(static_cast<long>(self->dataloader.batch_num()));
  HT_PY_FUNC_END
}

PyObject* PyDataloader_batch_size(PyDataloader* self) {
  HT_PY_FUNC_BEGIN
  return PyLong_FromLong(static_cast<long>(self->dataloader.batch_size()));
  HT_PY_FUNC_END
}

PyObject* PyDataloader_get_data(PyDataloader* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  Graph::push_graph_ctx(Graph::get_default_eager_graph().id());
  auto tensor = self->dataloader.get_arr();
  if (tensor == Tensor()) {
    PyErr_SetNone(PyExc_StopIteration);
    Graph::pop_graph_ctx();
    return NULL;
  }
  auto pytensor = PyTensor_New(std::move(tensor));
  Graph::pop_graph_ctx();
  return pytensor;
  HT_PY_FUNC_END
}

PyObject* PyDataloader_get_next_data(PyDataloader* self, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "get_next_data()"
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    return PyTensor_New(self->dataloader.get_next_arr());
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

// PyObject* PyDataloader_zerograd(PyDataloader* self) {
//   HT_PY_FUNC_BEGIN
//   self->dataloader.ZeroGrad();
//   HT_PY_FUNC_END
// }

// PyObject* PyDataloader_step(PyDataloader* self) {
//   HT_PY_FUNC_BEGIN
//   self->dataloader.Step();
//   HT_PY_FUNC_END
// }

// PyObject* PyDataloader_makestates(PyDataloader* self, PyObject* args, PyObject* kwargs) {
//   HT_PY_FUNC_BEGIN
//   static PyArgParser parser({
//     "makestates(Tensor variable, std::string state_name=\"\")"
//   });
//   auto parsed_args = parser.parse(args, kwargs);
//   if (parsed_args.signature_index() == 0) {
//     self->dataloader.MakeStates(parsed_args.get_tensor_optional(0),
//                                parsed_args.get_string_or_default(1));
//   } else {
//     HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
//     __builtin_unreachable();
//   }
//   HT_PY_FUNC_END
// }

// NOLINTNEXTLINE
PyGetSetDef PyDataloader_properties[] = {
  {PY_GET_SET_DEF_NAME("num_workers"), (getter) PyDataloader_num_workers, nullptr, nullptr, nullptr}, 
  {PY_GET_SET_DEF_NAME("batch_num"), (getter) PyDataloader_batch_num, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("batch_size"), (getter) PyDataloader_batch_size, nullptr, nullptr, nullptr},
  {PY_GET_SET_DEF_NAME("sample_num"), (getter) PyDataloader_sample_num, nullptr, nullptr, nullptr}, 
  {nullptr}
};

PyTypeObject PyDataloader_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hetu.Dataloader", /* tp_name */
  sizeof(PyDataloader), /* tp_basicsize */
  0, /* tp_itemsize */
  (destructor) PyDataloader_dealloc, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  (reprfunc) PyDataloader_repr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  (reprfunc) PyDataloader_str, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  (iternextfunc) PyDataloader_get_data, /* tp_iternext */
  nullptr, /* tp_methods */
  nullptr, /* tp_members */
  PyDataloader_properties, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  PyDataloader_pynew, /* tp_new */
};
PyTypeObject* PyDataloader_Type = &PyDataloader_Type_obj;

std::vector<PyMethodDef> InitDataloaderPyMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"get_data", (PyCFunction) PyDataloader_get_data, METH_VARARGS | METH_KEYWORDS, nullptr },
    {"get_next_data", (PyCFunction) PyDataloader_get_next_data, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    // {"makestates", (PyCFunction) PyDataloader_makestates, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    // {"zerograd", (PyCFunction) PyDataloader_zerograd, METH_NOARGS, nullptr }, 
    // {"step", (PyCFunction) PyDataloader_step, METH_NOARGS, nullptr },
    // {"_make_subclass", (PyCFunction) PyDataloader_make_subclass, METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_dataloader_methods());
  return ret;
}

std::vector<PyMethodDef> InitDataloaderPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    // {"from_numpy", (PyCFunction) PyDataloader_from_numpy, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {nullptr}
  });
  AddPyMethodDefs(ret, hetu::graph::get_registered_dataloader_class_methods());
  return ret;
}

void AddPyDataloaderTypeToModule(py::module_& module) {
  PyDataloader_Type->tp_as_number = &(get_registered_dataloader_number_methods());
  static auto dataloader_methods = InitDataloaderPyMethodDefs();
  PyDataloader_Type->tp_methods = dataloader_methods.data();
  HT_RUNTIME_ERROR_IF(PyType_Ready(PyDataloader_Type) < 0) 
    << "PyDataloader_Type not ready";
  Py_INCREF(PyDataloader_Type);
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddObject(
      module.ptr(), "Dataloader", reinterpret_cast<PyObject*>(PyDataloader_Type)))
    << "Failed to add PyDataloader_Type";
  
  static auto dataloader_class_methods = InitDataloaderPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), dataloader_class_methods.data()))
    << "Failed to add Dataloader class methods";
}

} // namespace graph
} // namespace hetu

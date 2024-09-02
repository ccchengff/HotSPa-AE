#include "hetu/_binding/utils/function_registry.h"

namespace hetu {

PyNumberMethods& get_registered_int_symbol_number_methods() {
  static PyNumberMethods PyIntSymbol_nums = {
    nullptr, /* nb_add */
    nullptr, /* nb_subtract */
    nullptr, /* nb_multiply */
    nullptr, /* nb_remainder */
    nullptr, /* nb_divmod */
    nullptr, /* nb_power */
    nullptr, /* nb_negative */
    nullptr, /* nb_positive */
    nullptr, /* nb_absolute */
    nullptr, /* nb_bool */
    nullptr, /* nb_invert */
    nullptr, /* nb_lshift */
    nullptr, /* nb_rshift */
    nullptr, /* nb_and */
    nullptr, /* nb_xor */
    nullptr, /* nb_or */
    nullptr, /* nb_int */
    nullptr, /* nb_reserved */
    nullptr, /* nb_float */

    nullptr, /* nb_inplace_add */
    nullptr, /* nb_inplace_subtract */
    nullptr, /* nb_inplace_multiply */
    nullptr, /* nb_inplace_remainder */
    nullptr, /* nb_inplace_power */
    nullptr, /* nb_inplace_lshift */
    nullptr, /* nb_inplace_rshift */
    nullptr, /* nb_inplace_and */
    nullptr, /* nb_inplace_xor */
    nullptr, /* nb_inplace_or */

    nullptr, /* nb_floor_divide */
    nullptr, /* nb_true_divide */
    nullptr, /* nb_inplace_floor_divide */
    nullptr, /* nb_inplace_true_divide */

    nullptr, /* nb_index */

    nullptr, /* nb_matrix_multiply */
    nullptr, /* nb_inplace_matrix_multiply */
  };
  return PyIntSymbol_nums;
}

std::vector<PyMethodDef>& get_registered_int_symbol_methods() {
  static std::vector<PyMethodDef> registered_int_symbol_methods = {{nullptr}};
  return registered_int_symbol_methods;
}

int RegisterIntSymbolMethod(const char* name, PyCFunction func, int flags,
                         const char* doc) {
  AddPyMethodDef(get_registered_int_symbol_methods(), {name, func, flags, doc});
  return 0;
}

std::vector<PyMethodDef>& get_registered_int_symbol_class_methods() {
  static std::vector<PyMethodDef> registered_int_symbol_class_methods = {{nullptr}};
  return registered_int_symbol_class_methods;
}

int RegisterIntSymbolClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc) {
  AddPyMethodDef(get_registered_int_symbol_class_methods(), {name, func, flags, doc});
  return 0;
}

namespace impl {

std::vector<PyMethodDef>& get_registered_ndarray_methods() {
  static std::vector<PyMethodDef> registered_ndarray_methods = {{nullptr}};
  return registered_ndarray_methods;
}

std::vector<PyMethodDef>& get_registered_ndarray_class_methods() {
  static std::vector<PyMethodDef> registered_ndarray_class_methods = {{nullptr}};
  return registered_ndarray_class_methods;
}

int RegisterNDArrayMethod(const char* name, PyCFunction func, int flags,
                          const char* doc) {
  AddPyMethodDef(get_registered_ndarray_methods(), {name, func, flags, doc});
  return 0;
}

int RegisterNDArrayClassMethod(const char* name, PyCFunction func, int flags,
                               const char* doc) {
  AddPyMethodDef(get_registered_ndarray_methods(), {name, func, flags, doc});
  return 0;
}

} // namespace impl

namespace graph {

PyNumberMethods& get_registered_tensor_number_methods() {
  static PyNumberMethods PyTensor_nums = {
    nullptr, /* nb_add */
    nullptr, /* nb_subtract */
    nullptr, /* nb_multiply */
    nullptr, /* nb_remainder */
    nullptr, /* nb_divmod */
    nullptr, /* nb_power */
    nullptr, /* nb_negative */
    nullptr, /* nb_positive */
    nullptr, /* nb_absolute */
    nullptr, /* nb_bool */
    nullptr, /* nb_invert */
    nullptr, /* nb_lshift */
    nullptr, /* nb_rshift */
    nullptr, /* nb_and */
    nullptr, /* nb_xor */
    nullptr, /* nb_or */
    nullptr, /* nb_int */
    nullptr, /* nb_reserved */
    nullptr, /* nb_float */

    nullptr, /* nb_inplace_add */
    nullptr, /* nb_inplace_subtract */
    nullptr, /* nb_inplace_multiply */
    nullptr, /* nb_inplace_remainder */
    nullptr, /* nb_inplace_power */
    nullptr, /* nb_inplace_lshift */
    nullptr, /* nb_inplace_rshift */
    nullptr, /* nb_inplace_and */
    nullptr, /* nb_inplace_xor */
    nullptr, /* nb_inplace_or */

    nullptr, /* nb_floor_divide */
    nullptr, /* nb_true_divide */
    nullptr, /* nb_inplace_floor_divide */
    nullptr, /* nb_inplace_true_divide */

    nullptr, /* nb_index */

    nullptr, /* nb_matrix_multiply */
    nullptr, /* nb_inplace_matrix_multiply */
  };
  return PyTensor_nums;
}

std::vector<PyMethodDef>& get_registered_tensor_methods() {
  static std::vector<PyMethodDef> registered_tensor_methods = {{nullptr}};
  return registered_tensor_methods;
}

int RegisterTensorMethod(const char* name, PyCFunction func, int flags,
                         const char* doc) {
  AddPyMethodDef(get_registered_tensor_methods(), {name, func, flags, doc});
  return 0;
}

std::vector<PyMethodDef>& get_registered_tensor_class_methods() {
  static std::vector<PyMethodDef> registered_tensor_class_methods = {{nullptr}};
  return registered_tensor_class_methods;
}

int RegisterTensorClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc) {
  AddPyMethodDef(get_registered_tensor_class_methods(), {name, func, flags, doc});
  return 0;
}

PyNumberMethods& get_registered_optimizer_number_methods() {
  static PyNumberMethods PyOptimizer_nums = {
    nullptr, /* nb_add */
    nullptr, /* nb_subtract */
    nullptr, /* nb_multiply */
    nullptr, /* nb_remainder */
    nullptr, /* nb_divmod */
    nullptr, /* nb_power */
    nullptr, /* nb_negative */
    nullptr, /* nb_positive */
    nullptr, /* nb_absolute */
    nullptr, /* nb_bool */
    nullptr, /* nb_invert */
    nullptr, /* nb_lshift */
    nullptr, /* nb_rshift */
    nullptr, /* nb_and */
    nullptr, /* nb_xor */
    nullptr, /* nb_or */
    nullptr, /* nb_int */
    nullptr, /* nb_reserved */
    nullptr, /* nb_float */

    nullptr, /* nb_inplace_add */
    nullptr, /* nb_inplace_subtract */
    nullptr, /* nb_inplace_multiply */
    nullptr, /* nb_inplace_remainder */
    nullptr, /* nb_inplace_power */
    nullptr, /* nb_inplace_lshift */
    nullptr, /* nb_inplace_rshift */
    nullptr, /* nb_inplace_and */
    nullptr, /* nb_inplace_xor */
    nullptr, /* nb_inplace_or */

    nullptr, /* nb_floor_divide */
    nullptr, /* nb_true_divide */
    nullptr, /* nb_inplace_floor_divide */
    nullptr, /* nb_inplace_true_divide */

    nullptr, /* nb_index */

    nullptr, /* nb_matrix_multiply */
    nullptr, /* nb_inplace_matrix_multiply */
  };
  return PyOptimizer_nums;
}

std::vector<PyMethodDef>& get_registered_optimizer_methods() {
  static std::vector<PyMethodDef> registered_optimizer_methods = {{nullptr}};
  return registered_optimizer_methods;
}

int RegisterOptimizerMethod(const char* name, PyCFunction func, int flags,
                            const char* doc) {
  AddPyMethodDef(get_registered_optimizer_methods(), {name, func, flags, doc});
  return 0;
}

std::vector<PyMethodDef>& get_registered_optimizer_class_methods() {
  static std::vector<PyMethodDef> registered_optimizer_class_methods = {{nullptr}};
  return registered_optimizer_class_methods;
}

int RegisterOptimizerClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc) {
  AddPyMethodDef(get_registered_optimizer_class_methods(), {name, func, flags, doc});
  return 0;
}
int RegisterInitializerMethod(const char* name, PyCFunction func, int flags,
                              const char* doc) {
  AddPyMethodDef(get_registered_initializer_methods(), {name, func, flags, doc});
  return 0;
}

std::vector<PyMethodDef>& get_registered_initializer_class_methods() {
  static std::vector<PyMethodDef> registered_initializer_class_methods = {{nullptr}};
  return registered_initializer_class_methods;
}

int RegisterInitializerClassMethod(const char* name, PyCFunction func, int flags,
                                   const char* doc) {
  AddPyMethodDef(get_registered_initializer_class_methods(), {name, func, flags, doc});
  return 0;
}

PyNumberMethods& get_registered_dataloader_number_methods() {
  static PyNumberMethods PyDataloader_nums = {
    nullptr, /* nb_add */
    nullptr, /* nb_subtract */
    nullptr, /* nb_multiply */
    nullptr, /* nb_remainder */
    nullptr, /* nb_divmod */
    nullptr, /* nb_power */
    nullptr, /* nb_negative */
    nullptr, /* nb_positive */
    nullptr, /* nb_absolute */
    nullptr, /* nb_bool */
    nullptr, /* nb_invert */
    nullptr, /* nb_lshift */
    nullptr, /* nb_rshift */
    nullptr, /* nb_and */
    nullptr, /* nb_xor */
    nullptr, /* nb_or */
    nullptr, /* nb_int */
    nullptr, /* nb_reserved */
    nullptr, /* nb_float */

    nullptr, /* nb_inplace_add */
    nullptr, /* nb_inplace_subtract */
    nullptr, /* nb_inplace_multiply */
    nullptr, /* nb_inplace_remainder */
    nullptr, /* nb_inplace_power */
    nullptr, /* nb_inplace_lshift */
    nullptr, /* nb_inplace_rshift */
    nullptr, /* nb_inplace_and */
    nullptr, /* nb_inplace_xor */
    nullptr, /* nb_inplace_or */

    nullptr, /* nb_floor_divide */
    nullptr, /* nb_true_divide */
    nullptr, /* nb_inplace_floor_divide */
    nullptr, /* nb_inplace_true_divide */

    nullptr, /* nb_index */

    nullptr, /* nb_matrix_multiply */
    nullptr, /* nb_inplace_matrix_multiply */
  };
  return PyDataloader_nums;
}

std::vector<PyMethodDef>& get_registered_dataloader_methods() {
  static std::vector<PyMethodDef> registered_dataloader_methods = {{nullptr}};
  return registered_dataloader_methods;
}

int RegisterDataloaderMethod(const char* name, PyCFunction func, int flags,
                             const char* doc) {
  AddPyMethodDef(get_registered_dataloader_methods(), {name, func, flags, doc});
  return 0;
}

std::vector<PyMethodDef>& get_registered_dataloader_class_methods() {
  static std::vector<PyMethodDef> registered_dataloader_class_methods = {{nullptr}};
  return registered_dataloader_class_methods;
}

int RegisterDataloaderClassMethod(const char* name, PyCFunction func, int flags,
                                  const char* doc) {
  AddPyMethodDef(get_registered_dataloader_class_methods(), {name, func, flags, doc});
}

std::vector<PyMethodDef>& get_registered_initializer_methods() {
  static std::vector<PyMethodDef> registered_initializer_methods = {{nullptr}};
  return registered_initializer_methods;
}


} // namespace graph

} // namespace hetu

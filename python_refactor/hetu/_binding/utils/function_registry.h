#pragma once

#include <Python.h>
#include "hetu/common/macros.h"
#include "hetu/_binding/utils/decl_utils.h"

namespace hetu {

PyNumberMethods& get_registered_int_symbol_number_methods();

std::vector<PyMethodDef>& get_registered_int_symbol_methods();

std::vector<PyMethodDef>& get_registered_int_symbol_class_methods();

int RegisterIntSymbolMethod(const char* name, PyCFunction func, int flags,
                         const char* doc);

int RegisterIntSymbolClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc);

#define REGISTER_INT_SYMBOL_NUMBER_METHOD(slot, func)                              \
  static auto __int_symbol_number_method_##slot##_registry =                       \
    ((hetu::get_registered_int_symbol_number_methods().slot) = (func))

#define REGISTER_INT_SYMBOL_METHOD(name, func, flags, doc)                         \
  static auto __int_symbol_method_##name##_registry =                              \
    hetu::RegisterIntSymbolMethod(quote(name), func, flags, doc)

#define REGISTER_INT_SYMBOL_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __int_symbol_class_method_##name##_registry =                        \
    hetu::RegisterIntSymbolClassMethod(quote(name), func, flags, doc)

namespace impl {

std::vector<PyMethodDef>& get_registered_ndarray_methods();

std::vector<PyMethodDef>& get_registered_ndarray_class_methods();

int RegisterNDArrayMethod(const char* name, PyCFunction func, int flags,
                          const char* doc);

int RegisterNDArrayClassMethod(const char* name, PyCFunction func, int flags,
                               const char* doc);

#define REGISTER_NDARRAY_METHOD(name, func, flags, doc)                        \
  static int __ndarray_method_##name##_registry =                              \
    hetu::impl::RegisterNDArrayMethod(quote(name), func, flags, doc)

#define REGISTER_NDARRAY_CLASS_METHOD(name, func, flags, doc)                  \
  static int __ndarray_class_method_##name##_registry =                        \
    hetu::impl::RegisterNDArrayClassMethod(quote(name), func, flags, doc)

} // namespace impl

namespace graph {

PyNumberMethods& get_registered_tensor_number_methods();

std::vector<PyMethodDef>& get_registered_tensor_methods();

std::vector<PyMethodDef>& get_registered_tensor_class_methods();

int RegisterTensorMethod(const char* name, PyCFunction func, int flags,
                         const char* doc);

int RegisterTensorClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc);

#define REGISTER_TENSOR_NUMBER_METHOD(slot, func)                              \
  static auto __tensor_number_method_##slot##_registry =                       \
    ((hetu::graph::get_registered_tensor_number_methods().slot) = (func))

#define REGISTER_TENSOR_METHOD(name, func, flags, doc)                         \
  static auto __tensor_method_##name##_registry =                              \
    hetu::graph::RegisterTensorMethod(quote(name), func, flags, doc)

#define REGISTER_TENSOR_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __tensor_class_method_##name##_registry =                        \
    hetu::graph::RegisterTensorClassMethod(quote(name), func, flags, doc)

PyNumberMethods& get_registered_optimizer_number_methods();

std::vector<PyMethodDef>& get_registered_optimizer_methods();

std::vector<PyMethodDef>& get_registered_optimizer_class_methods();

int RegisterOptimizerMethod(const char* name, PyCFunction func, int flags,
                            const char* doc);

int RegisterOptimizerClassMethod(const char* name, PyCFunction func, int flags,
                                 const char* doc);

#define REGISTER_OPTIMIZER_NUMBER_METHOD(slot, func)                              \
  static auto __optimizer_number_method_##slot##_registry =                       \
    ((hetu::graph::get_registered_optimizer_number_methods().slot) = (func))

#define REGISTER_OPTIMIZER_METHOD(name, func, flags, doc)                         \
  static auto __optimizer_method_##name##_registry =                              \
    hetu::graph::RegisterOptimizerMethod(quote(name), func, flags, doc)

#define REGISTER_OPTIMIZER_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __optimizer_class_method_##name##_registry =                        \
    hetu::graph::RegisterOptimizerClassMethod(quote(name), func, flags, doc)

int RegisterInitializerMethod(const char* name, PyCFunction func, int flags,
                              const char* doc);

int RegisterInitializerClassMethod(const char* name, PyCFunction func, int flags,
                                   const char* doc);

#define REGISTER_INITIALIZER_METHOD(name, func, flags, doc)                         \
  static auto __initializer_method_##name##_registry =                              \
    hetu::graph::RegisterInitializerMethod(quote(name), func, flags, doc)

#define REGISTER_INITIALIZER_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __initializer_class_method_##name##_registry =                        \
    hetu::graph::RegisterInitializerClassMethod(quote(name), func, flags, doc)    

PyNumberMethods& get_registered_dataloader_number_methods();

std::vector<PyMethodDef>& get_registered_dataloader_methods();

std::vector<PyMethodDef>& get_registered_dataloader_class_methods();

int RegisterDataloaderMethod(const char* name, PyCFunction func, int flags,
                            const char* doc);

int RegisterDataloaderClassMethod(const char* name, PyCFunction func, int flags,
                                 const char* doc);

#define REGISTER_DATALOADER_NUMBER_METHOD(slot, func)                              \
  static auto __dataloader_number_method_##slot##_registry =                       \
    ((hetu::graph::get_registered_dataloader_number_methods().slot) = (func))

#define REGISTER_DATALOADER_METHOD(name, func, flags, doc)                         \
  static auto __dataloader_method_##name##_registry =                              \
    hetu::graph::RegisterDataloaderMethod(quote(name), func, flags, doc)

#define REGISTER_DATALOADER_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __dataloader_class_method_##name##_registry =                        \
    hetu::graph::RegisterDataloaderClassMethod(quote(name), func, flags, doc)
std::vector<PyMethodDef>& get_registered_initializer_methods();

std::vector<PyMethodDef>& get_registered_initializer_class_methods();

} // namespace graph

} // namespace hetu

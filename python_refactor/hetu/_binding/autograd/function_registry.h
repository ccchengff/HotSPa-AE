#pragma once

#include <Python.h>
#include "hetu/common/macros.h"
#include "hetu/_binding/utils/decl_utils.h"

namespace hetu {

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

namespace autograd {

PyNumberMethods& get_registered_tensor_number_methods();

std::vector<PyMethodDef>& get_registered_tensor_methods();

std::vector<PyMethodDef>& get_registered_tensor_class_methods();

int RegisterTensorMethod(const char* name, PyCFunction func, int flags,
                         const char* doc);

int RegisterTensorClassMethod(const char* name, PyCFunction func, int flags,
                              const char* doc);

#define REGISTER_TENSOR_NUMBER_METHOD(slot, func)                              \
  static auto __tensor_number_method_##slot##_registry =                       \
    ((hetu::autograd::get_registered_tensor_number_methods().slot) = (func))

#define REGISTER_TENSOR_METHOD(name, func, flags, doc)                         \
  static auto __tensor_method_##name##_registry =                              \
    hetu::autograd::RegisterTensorMethod(quote(name), func, flags, doc)

#define REGISTER_TENSOR_CLASS_METHOD(name, func, flags, doc)                   \
  static auto __tensor_class_method_##name##_registry =                        \
    hetu::autograd::RegisterTensorClassMethod(quote(name), func, flags, doc)

} // namespace autograd

} // namespace hetu

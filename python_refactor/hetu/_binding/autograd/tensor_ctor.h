#pragma once

#include <Python.h>
#include "hetu/_binding/autograd/tensor.h"
#include "hetu/autograd/init/initializer.h"

namespace hetu {
namespace autograd {

PyObject* TensorCopyCtor(PyTypeObject* type, PyObject* args, PyObject* kwargs);

} // namespace autograd
} // namespace hetu

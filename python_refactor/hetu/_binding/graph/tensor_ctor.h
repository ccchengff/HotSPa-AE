#pragma once

#include <Python.h>
#include "hetu/_binding/graph/tensor.h"
#include "hetu/graph/init/initializer.h"

namespace hetu {
namespace graph {

PyObject* TensorCopyCtor(PyTypeObject* type, PyObject* args, PyObject* kwargs);

} // namespace graph
} // namespace hetu

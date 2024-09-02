#pragma once

#include <Python.h>
#include <mpi.h>
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/core/stream.h"
#include "hetu/_binding/utils/pybind_common.h"

namespace hetu {

struct PyCommGroup {
  PyObject_HEAD;
};

extern PyTypeObject* PyCommGroup_Type;

void AddPyCommGroupTypeToModule(py::module_& module);

} // namespace hetu

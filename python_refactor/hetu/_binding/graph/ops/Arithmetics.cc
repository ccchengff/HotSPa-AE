#include "hetu/_binding/graph/ops/python_headers.h"

namespace hetu {
namespace graph {

PyObject* PyTensor_add(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "add(Tensor x, Tensor y)",
    "add(Tensor x, float y)",
    "add(float x, Tensor y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(MakeAddElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor,
      reinterpret_cast<PyTensor*>(y)->tensor, CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(
      MakeAddByConstOp(reinterpret_cast<PyTensor*>(x)->tensor,
                       Float64_FromPyFloat(y), CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(MakeAddByConstOp(
      Float64_FromPyFloat(x), reinterpret_cast<PyTensor*>(y)->tensor,
      CurrentOpMetaCtx()));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_NUMBER_METHOD(nb_add, (binaryfunc) PyTensor_add);

PyObject* PyTensor_sub(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "sub(Tensor x, Tensor y)",
    "sub(Tensor x, float y)",
    "sub(float x, Tensor y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(MakeSubElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor,
      reinterpret_cast<PyTensor*>(y)->tensor, CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(
      MakeSubByConstOp(reinterpret_cast<PyTensor*>(x)->tensor,
                       Float64_FromPyFloat(y), CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(MakeSubFromConstOp(
      Float64_FromPyFloat(x), reinterpret_cast<PyTensor*>(y)->tensor,
      CurrentOpMetaCtx()));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_NUMBER_METHOD(nb_subtract, (binaryfunc) PyTensor_sub);

PyObject* PyTensor_neg(PyObject* x) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"neg(Tensor x)"});
  auto parsed_args = parser.parse(x, nullptr, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(
      MakeNegateOp(reinterpret_cast<PyTensor*>(x)->tensor, CurrentOpMetaCtx()));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_NUMBER_METHOD(nb_negative, (unaryfunc) PyTensor_neg);

PyObject* PyTensor_mul(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "mul(Tensor x, Tensor y)",
    "mul(Tensor x, float y)",
    "mul(float x, Tensor y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(MakeMulElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor,
      reinterpret_cast<PyTensor*>(y)->tensor, CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(
      MakeMulByConstOp(reinterpret_cast<PyTensor*>(x)->tensor,
                       Float64_FromPyFloat(y), CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(MakeMulByConstOp(
      Float64_FromPyFloat(x), reinterpret_cast<PyTensor*>(y)->tensor,
      CurrentOpMetaCtx()));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_NUMBER_METHOD(nb_multiply, (binaryfunc) PyTensor_mul);

PyObject* PyTensor_div(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "div(Tensor x, Tensor y)",
    "div(Tensor x, float y)",
    "div(float x, Tensor y)",
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(MakeDivElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor,
      reinterpret_cast<PyTensor*>(y)->tensor, CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(
      MakeDivByConstOp(reinterpret_cast<PyTensor*>(x)->tensor,
                       Float64_FromPyFloat(y), CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(MakeDivFromConstOp(
      Float64_FromPyFloat(x), reinterpret_cast<PyTensor*>(y)->tensor,
      CurrentOpMetaCtx()));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_NUMBER_METHOD(nb_true_divide, (binaryfunc) PyTensor_div);

} // namespace graph
} // namespace hetu

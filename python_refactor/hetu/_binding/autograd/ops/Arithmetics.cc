#include "hetu/_binding/autograd/ops/python_headers.h"

namespace hetu {
namespace autograd {

PyObject* PyTensor_add(PyObject* x, PyObject* y) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "add(Tensor x, Tensor y)", 
    "add(Tensor x, float y)", 
    "add(float x, Tensor y)", 
  });
  auto parsed_args = parser.parse(x, y, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(AddElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      reinterpret_cast<PyTensor*>(y)->tensor, 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(AddByConstOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      Float64_FromPyFloat(y), 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(AddByConstOp(
      Float64_FromPyFloat(x), 
      reinterpret_cast<PyTensor*>(y)->tensor, 
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
    return PyObject_FromOperatorOutputs(SubElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      reinterpret_cast<PyTensor*>(y)->tensor, 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(SubByConstOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      Float64_FromPyFloat(y), 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(SubFromConstOp(
      Float64_FromPyFloat(x), 
      reinterpret_cast<PyTensor*>(y)->tensor, 
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
  static PyArgParser parser({
    "neg(Tensor x)"
  });
  auto parsed_args = parser.parse(x, nullptr, nullptr, true);
  if (parsed_args.signature_index() == 0) {
    return PyObject_FromOperatorOutputs(NegateOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      CurrentOpMetaCtx()));
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
    return PyObject_FromOperatorOutputs(MulElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      reinterpret_cast<PyTensor*>(y)->tensor, 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(MulByConstOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      Float64_FromPyFloat(y), 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(MulByConstOp(
      Float64_FromPyFloat(x), 
      reinterpret_cast<PyTensor*>(y)->tensor, 
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
    return PyObject_FromOperatorOutputs(DivElewiseOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      reinterpret_cast<PyTensor*>(y)->tensor, 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 1) {
    return PyObject_FromOperatorOutputs(DivByConstOp(
      reinterpret_cast<PyTensor*>(x)->tensor, 
      Float64_FromPyFloat(y), 
      CurrentOpMetaCtx()));
  } else if (parsed_args.signature_index() == 2) {
    return PyObject_FromOperatorOutputs(DivFromConstOp(
      Float64_FromPyFloat(x), 
      reinterpret_cast<PyTensor*>(y)->tensor, 
      CurrentOpMetaCtx()));
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  HT_PY_FUNC_END
}

REGISTER_TENSOR_NUMBER_METHOD(nb_true_divide, (binaryfunc) PyTensor_div);

} // namespace autograd
} // namespace hetu

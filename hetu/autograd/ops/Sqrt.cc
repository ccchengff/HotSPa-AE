#include "hetu/autograd/ops/Sqrt.h"
#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SqrtOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Sqrt,
                                  inputs.at(0), outputs.at(0), stream());
}

TensorList SqrtOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input =
    MulByConstOp(
      MulElewiseOp(ReciprocalSqrtOp(_inputs[0], g_op_meta)->output(0),
                   grad_outputs.at(0), g_op_meta)
        ->output(0),
      0.5, g_op_meta.set_name(grad_name()))
      ->output(0);
  return {grad_input};
}

void SqrtOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList SqrtOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void ReciprocalSqrtOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ReciprocalSqrt, inputs.at(0),
                                  outputs.at(0), stream());
}

void ReciprocalSqrtOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ReciprocalSqrtOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

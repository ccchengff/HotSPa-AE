#include "hetu/autograd/ops/Tanh.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void TanhOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Tanh,
                                  inputs.at(0), outputs.at(0), stream());
}

TensorList TanhOpDef::DoGradient(const TensorList& grad_outputs) {
  return {TanhGradientOp(_outputs[0], grad_outputs.at(0),
                         grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void TanhOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList TanhOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void TanhGradientOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::TanhGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), stream());
}

void TanhGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList TanhGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

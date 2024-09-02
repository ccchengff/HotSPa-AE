#include "hetu/autograd/ops/Relu.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ReluOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Relu,
                                  inputs.at(0), outputs.at(0), stream());
}

TensorList ReluOpDef::DoGradient(const TensorList& grad_outputs) {
  return {ReluGradientOp(_inputs[0], grad_outputs.at(0),
                         grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void ReluOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ReluOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void ReluGradientOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ReluGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), stream());
}

void ReluGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ReluGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

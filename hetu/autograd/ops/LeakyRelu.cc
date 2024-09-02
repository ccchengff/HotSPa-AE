#include "hetu/autograd/ops/LeakyRelu.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void LeakyReluOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::LeakyRelu, inputs.at(0),
                                  get_alpha(), outputs.at(0), stream());
}

TensorList LeakyReluOpDef::DoGradient(const TensorList& grad_outputs) {
  return {LeakyReluGradientOp(_inputs[0], grad_outputs.at(0), get_alpha(),
                              grad_op_meta().set_name(grad_name(0)))
            ->output(0)};
}

void LeakyReluOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList LeakyReluOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void LeakyReluGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::LeakyReluGradient, inputs.at(0),
    inputs.at(1), get_alpha(), outputs.at(0), stream());
}

void LeakyReluGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList
LeakyReluGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

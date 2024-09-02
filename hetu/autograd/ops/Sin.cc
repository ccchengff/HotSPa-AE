#include "hetu/autograd/ops/Sin.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SinOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Sin,
                                  inputs.at(0), outputs.at(0), stream());
}

TensorList SinOpDef::DoGradient(const TensorList& grad_outputs) {
  return {CosOp(_inputs[0], 
          grad_op_meta().set_name(grad_name()))->output(0)};
}

void SinOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList SinOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void CosOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Cos, inputs.at(0),
                                  outputs.at(0), stream());
}

void CosOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList CosOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

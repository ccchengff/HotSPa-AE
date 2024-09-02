#include "hetu/autograd/ops/Dropout.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace autograd {

NDArrayList DropoutOpDef::DoCompute(const NDArrayList& inputs,
                                    RuntimeContext& ctx) {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(inputs, ctx);
  if (recompute()) {
    uint64_t seed = hetu::impl::GenNextRandomSeed();
    ctx.get_op_ctx(id()).put_uint64("seed", seed);
    HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                                 hetu::impl::Dropout, inputs.at(0),
                                 1 - keep_prob(), seed, outputs[0], stream());
    return outputs;
  } else {
    HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                                 hetu::impl::Dropout, inputs.at(0),
                                 1 - keep_prob(), 0, outputs[0], stream());
    return outputs;
  }
}

TensorList DropoutOpDef::DoGradient(const TensorList& grad_outputs) {
  if (recompute()) {
    return {DropoutGradientWithRecomputationOp(
              grad_outputs.at(0), id(), keep_prob(), grad_op_meta().set_name(grad_name()))
              ->output(0)};
  } else {
    return {DropoutGradientOp(grad_outputs.at(0), output(0), keep_prob(),
                              grad_op_meta().set_name(grad_name()))
              ->output(0)};
  }
}

void DropoutOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList DropoutOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

NDArrayList DropoutGradientOpDef::DoCompute(const NDArrayList& inputs,
                                            RuntimeContext& ctx) {
  NDArrayList outputs = DoAllocOutputs(inputs, ctx);
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::DropoutGradient, inputs.at(0),
    inputs.at(1), 1 - keep_prob(), outputs[0], stream());
  return outputs;
}

void DropoutGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList
DropoutGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

NDArrayList
DropoutGradientWithRecomputationOpDef::DoCompute(const NDArrayList& inputs,
                                                 RuntimeContext& ctx) {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(inputs, ctx);
  uint64_t seed = ctx.get_op_ctx(_forward_op).get_uint64("seed");
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::DropoutGradientWithRecomputation,
    inputs.at(0), 1 - keep_prob(), seed, outputs[0], stream());
  return outputs;
}

void DropoutGradientWithRecomputationOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList DropoutGradientWithRecomputationOpDef::DoInferShape(
  const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

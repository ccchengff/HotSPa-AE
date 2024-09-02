#include "hetu/autograd/ops/Dropout2d.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace autograd {

NDArrayList Dropout2dOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  // TODO: support without recomputation
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(inputs, ctx);
  uint64_t seed = hetu::impl::GenNextRandomSeed();
  ctx.get_op_ctx(id()).put_uint64("seed", seed);
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                               hetu::impl::Dropout2d, inputs.at(0),
                               1 - keep_prob(), seed, outputs[0], stream());
  return outputs;
}

TensorList Dropout2dOpDef::DoGradient(const TensorList& grad_outputs) {
  auto& self = reinterpret_cast<Dropout2dOp&>(get_self());
  return {Dropout2dGradientWithRecomputationOp(
            grad_outputs.at(0), self, grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList Dropout2dOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void Dropout2dOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "Dropout2dOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) 
    << "Tensor input shouldn't be partial";
  HT_ASSERT(ds_input.check_max_dim(2))
    << "droupout2d only support split dimensions N&C in [N, C, H, W] now!";
  _outputs[0]->set_distributed_states(ds_input);  
}

NDArrayList
Dropout2dGradientWithRecomputationOpDef::DoCompute(const NDArrayList& inputs,
                                                   RuntimeContext& ctx) {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(inputs, ctx);
  uint64_t seed = ctx.get_op_ctx(_forward_op->id()).get_uint64("seed");
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::Dropout2dGradientWithRecomputation,
    inputs.at(0), 1 - keep_prob(), seed, outputs[0], stream());
  return outputs;
}

HTShapeList Dropout2dGradientWithRecomputationOpDef::DoInferShape(
  const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

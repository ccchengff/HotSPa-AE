#include "hetu/graph/ops/Dropout2d.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace graph {

void Dropout2dOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  uint64_t seed = hetu::impl::GenNextRandomSeed();
  ctx.get_or_create(op->id()).put_uint64("seed", seed);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                hetu::impl::Dropout2d, inputs.at(0), 1 - keep_prob(),
                                op->id(), outputs.at(0), op->instantiation_ctx().stream());
};

NDArrayList Dropout2dOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs,
                                       RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList Dropout2dOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {MakeDropout2dGradientWithRecomputationOp(
            grad_outputs.at(0), op->id(), keep_prob(), inplace(), op->grad_op_meta().set_name(op->grad_name()))};
}

void Dropout2dOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "Dropout2dOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) 
    << "Tensor input shouldn't be partial";
  HT_ASSERT(ds_input.check_max_dim(2))
    << "droupout2d only support split dimensions N&C in [N, C, H, W] now!";
  outputs.at(0)->set_distributed_states(ds_input);  
}

void Dropout2dGradientWithRecomputationOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                                         NDArrayList& outputs,
                                                         RuntimeContext& ctx) const {
  uint64_t seed = ctx.get(_forward_op).get_uint64("seed");
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Dropout2dGradientWithRecomputation,
    inputs.at(0), 1 - keep_prob(), seed, outputs[0], op->instantiation_ctx().stream());
};

NDArrayList
Dropout2dGradientWithRecomputationOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                                    RuntimeContext& ctx) const {
  NDArrayList outputs = fw_inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

Tensor MakeDropout2dOp(Tensor input, double keep_prob,
                       bool recompute, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Dropout2dOpImpl>(keep_prob, recompute, false),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropout2dInplaceOp(Tensor input, double keep_prob,
                              bool recompute, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Dropout2dOpImpl>(keep_prob, recompute, true),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropout2dGradientWithRecomputationOp(Tensor grad_output,
                                                OpId forward_op, double keep_prob,
                                                bool fw_inplace,
                                                OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Dropout2dGradientWithRecomputationOpImpl>(forward_op, keep_prob, fw_inplace),
          {std::move(grad_output)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu

#include "hetu/graph/ops/Dropout.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace graph {

void DropoutOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  if (recompute()) {
    uint64_t seed = hetu::impl::GenNextRandomSeed();
    ctx.get_or_create(op->id()).put_uint64("seed", seed);
    HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                 hetu::impl::Dropout, inputs.at(0), 1 - keep_prob(),
                                 op->id(), outputs.at(0), op->instantiation_ctx().stream());
  } else {
    HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                 hetu::impl::Dropout, inputs.at(0), 1 - keep_prob(),
                                 0, outputs.at(0), op->instantiation_ctx().stream());
  }
};

NDArrayList DropoutOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList DropoutOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  if (recompute()) {
    return {MakeDropoutGradientWithRecomputationOp(
              grad_outputs.at(0), op->id(), keep_prob(), inplace(), op->grad_op_meta().set_name(op->grad_name()))};
  } else {
    return {MakeDropoutGradientOp(grad_outputs.at(0), op->output(0), keep_prob(),
                                  op->grad_op_meta().set_name(op->grad_name()))};
  }
}

void DropoutGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutGradient, inputs.at(0),
    inputs.at(1), 1 - keep_prob(), outputs[0], op->instantiation_ctx().stream());
};

NDArrayList DropoutGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                            RuntimeContext& ctx) const {
  NDArrayList outputs = DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void DropoutGradientWithRecomputationOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                                       NDArrayList& outputs,
                                                       RuntimeContext& ctx) const {
  uint64_t seed = ctx.get(_forward_op).get_uint64("seed");
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutGradientWithRecomputation,
    inputs.at(0), 1 - keep_prob(), seed, outputs[0], op->instantiation_ctx().stream());
};

NDArrayList
DropoutGradientWithRecomputationOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                                 RuntimeContext& ctx) const {
  NDArrayList outputs = fw_inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

Tensor MakeDropoutOp(Tensor input, double keep_prob,
                     bool recompute, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutOpImpl>(keep_prob, recompute, false),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutInplaceOp(Tensor input, double keep_prob,
                            bool recompute, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutOpImpl>(keep_prob, recompute, true),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutGradientOp(Tensor grad_output, Tensor output, double keep_prob,
                             OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutGradientOpImpl>(keep_prob),
          {std::move(grad_output), std::move(output)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutGradientWithRecomputationOp(Tensor grad_output, OpId forward_op, double keep_prob,
                                              bool inplace, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutGradientWithRecomputationOpImpl>(forward_op, keep_prob, inplace),
          {std::move(grad_output)},
          std::move(op_meta))->output(0);
}


} // namespace graph
} // namespace hetu

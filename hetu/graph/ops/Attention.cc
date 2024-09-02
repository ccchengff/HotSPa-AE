#include "hetu/graph/ops/Attention.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void AttentionOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(inputs.at(0)->shape(3), -0.5);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::FlashAttn,
                               inputs.at(0), inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
                               outputs.at(2), outputs.at(3), outputs.at(4), outputs.at(5),
                               outputs.at(6), p_dropout(), softmax_scale_, is_causal(), 
                               return_softmax(), op->instantiation_ctx().stream());
}

TensorList AttentionOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  return op->requires_grad(0) ? MakeAttentionGradientOp(grad_outputs.at(0), op->input(0), op->input(1), 
                                                        op->input(2), op->output(0), op->output(5),
                                                        op->output(6), p_dropout(), softmax_scale(),
                                                        is_causal(), op->grad_op_meta().set_name(op->grad_name()))
                              : empty;
}

HTShapeList AttentionOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes, 
                                          RuntimeContext& ctx) const {
  HTShapeList out_shapes = {input_shapes.at(0)};
  const int batch_size = input_shapes.at(0)[0];
  const int seqlen_q = input_shapes.at(0)[1];
  const int num_heads = input_shapes.at(0)[2];
  const int head_size_og = input_shapes.at(0)[3];
  const int seqlen_k = input_shapes.at(1)[1];
  const int num_heads_k = input_shapes.at(1)[2];
  HT_ASSERT(batch_size > 0)
  << "batch size must be postive";
  HT_ASSERT(head_size_og <= 256)
  << "FlashAttention forward only supports head dimension at most 256";
  HT_ASSERT(num_heads % num_heads_k == 0)
  << "Number of heads in key/value must divide number of heads in query";
  const int pad_len = head_size_og % 8 == 0 ? 0 : 8 - head_size_og % 8;
  HTShape padded_shape;
  for (int i = 0; i < 3; ++i) {
    padded_shape = input_shapes.at(i);
    padded_shape[3] += pad_len;
    out_shapes.emplace_back(padded_shape); //q_padded, k_padded, v_padded.
  }
  padded_shape = input_shapes.at(0);
  padded_shape[3] += pad_len;
  out_shapes.emplace_back(padded_shape); //out_padded
  HTShape lse_shape = {batch_size, num_heads, seqlen_q},
          rng_shape = {2};
  out_shapes.emplace_back(lse_shape); //softmax_lse
  out_shapes.emplace_back(rng_shape); //rng_state
  return out_shapes;
}

void AttentionGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                        NDArrayList& outputs, RuntimeContext& ctx) const {
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(inputs.at(1)->shape(3), -0.5);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::FlashAttnGradient, inputs.at(0),
                               inputs.at(1), inputs.at(2), inputs.at(3), const_cast<NDArray&>(inputs.at(4)),
                               const_cast<NDArray&>(inputs.at(5)), const_cast<NDArray&>(inputs.at(6)), 
                               outputs.at(0), outputs.at(1), outputs.at(2), p_dropout(), softmax_scale_,
                               is_causal(), op->instantiation_ctx().stream());
}

HTShapeList AttentionGradientOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(1), input_shapes.at(2), input_shapes.at(3)};
}

TensorList MakeAttentionOp(Tensor q, Tensor k, Tensor v, double p_dropout, double softmax_scale, 
                           bool is_causal, bool return_softmax, OpMeta op_meta) {
  TensorList inputs = {std::move(q), std::move(k), std::move(v)};
  return Graph::MakeOp(
        std::make_shared<AttentionOpImpl>(p_dropout, softmax_scale, is_causal, return_softmax),
        std::move(inputs),
        std::move(op_meta))->outputs();
}

TensorList MakeAttentionGradientOp(Tensor grad_out, Tensor q, Tensor k, Tensor v,
                                   Tensor out, Tensor softmax_lse, Tensor rng_state,
                                   double p_dropout, double softmax_scale,
                                   bool is_causal, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<AttentionGradientOpImpl>(p_dropout, softmax_scale, is_causal),
        {std::move(grad_out), std::move(q), std::move(k), std::move(v),
         std::move(out), std::move(softmax_lse), std::move(rng_state)},
         std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu

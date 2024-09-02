#include "hetu/graph/ops/Concat.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ConcatOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Concat, inputs.at(0), inputs.at(1),
    outputs.at(0), get_axis(), op->instantiation_ctx().stream());
  // NDArray::cat(inputs, get_axis(), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ConcatOpImpl::DoGradient(Operator &op,
                                    const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_inputA = op->requires_grad(0) ? MakeConcatGradientOp(op->input(0), grad_outputs.at(0), get_axis(), 0,
                                           g_op_meta.set_name(op->grad_name(0)))
                                         : Tensor();
  auto grad_inputB = op->requires_grad(1) ? MakeConcatGradientOp(op->input(1), grad_outputs.at(0), get_axis(), 1,
                                           g_op_meta.set_name(op->grad_name(1)))
                                         : Tensor();
  return {grad_inputA, grad_inputB};
}

HTShapeList ConcatOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const { 
  HTShape shapeA = input_shapes.at(0);
  shapeA[get_axis()] += input_shapes.at(1)[get_axis()];
  return {shapeA};
}

void ConcatOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta) const {
  const DistributedStates& ds_a = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_b = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "ConcatOpDef: distributed states for input a and input b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a & b shouldn't be partial";  
  HT_ASSERT(ds_a.check_equal(ds_b)) 
    << "Distributed states for tensor a and tensor b must be equal!";
  HT_ASSERT(ds_a.get_dim(get_axis()) == 1)
    << "Concat was not allowed in splited dimension: " << get_axis();
  outputs.at(0)->set_distributed_states(ds_a);
}

void ConcatGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::ConcatGradient, 
    inputs.at(1), outputs.at(0), get_axis(),
    get_id(), op->instantiation_ctx().stream());
}

HTShapeList ConcatGradientOpImpl::DoInferShape(Operator& op,
                                               const HTShapeList& input_shapes,
                                               RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ConcatGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeConcatOp(Tensor inputA, Tensor inputB, size_t axis,
                    OpMeta op_meta) {
  TensorList inputs = {std::move(inputA), std::move(inputB)};
  return Graph::MakeOp(
          std::make_shared<ConcatOpImpl>(axis),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

Tensor MakeConcatGradientOp(Tensor input, Tensor grad_output, size_t axis, size_t id,
                            OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<ConcatGradientOpImpl>(axis, id),
          {std::move(input), std::move(grad_output)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu

#include "hetu/graph/ops/AsStrided.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

namespace {

inline int64_t min_storage_size(HTShape shape, HTStride stride, int64_t storage_offset) {
  int64_t storage_size = storage_offset + 1;
  int64_t ndim = shape.size();
  for (int64_t i = 0; i < ndim; i++) {
    const auto& size_i = shape[i];
    if (size_i == 0) {
      return storage_offset;
    }
    storage_size += (size_i - 1) * stride[i];
  }
  return storage_size;
}

} // namespace

NDArrayList AsStridedOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs,
                                       RuntimeContext& ctx) const {
  NDArray output = NDArray::as_strided(inputs.at(0), outshape(), stride(),
                                       storage_offset(),
                                       op->instantiation_ctx().stream_index);
  return {output};
}

TensorList AsStridedOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeAsStridedGradientOp(grad_outputs.at(0), op->input(0),
                                          outshape(), stride(), storage_offset(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList
AsStridedOpImpl::DoInferShape(Operator& op,
                              const HTShapeList& input_shapes,
                              RuntimeContext& ctx) const {
  return {outshape()};
}

void AsStridedOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "AsStridedOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Input tensor cannot be splited in any dimension!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

void AsStridedGradientOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs, NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  auto in_storage_offset = inputs.at(1)->storage_offset();
  auto out_storage_offset = storage_offset();
  auto shared_offset = std::min(in_storage_offset, out_storage_offset);
  auto in_offset = in_storage_offset - shared_offset;
  auto out_offset = out_storage_offset - shared_offset;
  HTShape in_shape = inputs.at(1)->shape();
  HTStride in_stride = inputs.at(1)->stride();

  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::AsStridedGradient, inputs.at(0),
                               outputs.at(0), outshape(), stride(), in_shape, in_stride,
                               in_offset, out_offset, op->instantiation_ctx().stream());
}

NDArrayList
AsStridedGradientOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   RuntimeContext& ctx) const {
  HTShape in_shape = inputs.at(1)->shape();
  HTStride in_stride = inputs.at(1)->stride();
  HTShape out_shape = outshape();
  HTStride out_stride = stride();
  
  // Allocate grad_input storage
  auto in_storage_offset = inputs.at(1)->storage_offset();
  auto out_storage_offset = storage_offset();
  auto shared_offset = std::min(in_storage_offset, out_storage_offset);
  auto in_offset = in_storage_offset - shared_offset;
  auto out_offset = out_storage_offset - shared_offset;
  auto base_size1 = min_storage_size(in_shape, in_stride, in_offset);
  auto base_size2 = min_storage_size(out_shape, out_stride, out_offset);
  auto base_size = std::max(base_size1, base_size2);
  NDArray grad_input = NDArray::zeros({base_size}, op->instantiation_ctx().placement,
                                      op->input(1)->dtype(), op->instantiation_ctx().stream_index);
  NDArrayList outputs = {grad_input};
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

HTShapeList
AsStridedGradientOpImpl::DoInferShape(Operator& op, 
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  return {input_shapes[1]};
}

void AsStridedGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
}

Tensor MakeAsStridedOp(Tensor input, const HTShape& outshape, const HTStride& stride,
                       int64_t storage_offset, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AsStridedOpImpl>(outshape, stride, storage_offset),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAsStridedGradientOp(Tensor grad_output, Tensor input, const HTShape& outshape,
                               const HTStride& stride, int64_t storage_offset, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AsStridedGradientOpImpl>(outshape, stride, storage_offset),
           {std::move(grad_output), std::move(input)},
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu

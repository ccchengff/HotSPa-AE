#include "hetu/graph/headers.h"
#include "hetu/graph/ops/update_scale.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {
void UpdateScaleOpImpl::DoCompute(Operator& op, 
                                  const NDArrayList& inputs, NDArrayList& outputs,
                                  RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::UpdateScale,
                               outputs.at(0), outputs.at(1), inputs.at(2), growth_factor(), backoff_factor(),  
                               growth_interval(), op->instantiation_ctx().stream());
}

TensorList UpdateScaleOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {Tensor()};
}

HTShapeList UpdateScaleOpImpl::DoInferShape(Operator& op, 
                                            const HTShapeList& input_shapes, 
                                            RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

TensorList MakeUpdateScaleOp(Tensor scale, Tensor growth_tracker, Tensor found_inf, double growth_factor,
                             double backoff_factor, int growth_interval, OpMeta op_meta) {
  TensorList inputs = {std::move(scale), std::move(growth_tracker), std::move(found_inf)};
  // DataType input_type = DataType::FLOAT32;
  // 
  return Graph::MakeOp(
         std::make_shared<UpdateScaleOpImpl>(growth_factor, backoff_factor, growth_interval),
         std::move(inputs),
         std::move(op_meta))->outputs();
}
} // namespace graph
} // namespace hetu

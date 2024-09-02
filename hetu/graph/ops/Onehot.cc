#include "hetu/graph/ops/Onehot.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void OnehotOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::Onehot, inputs.at(0),
                                  num_classes(), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList OnehotOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {Tensor()};
}

HTShapeList OnehotOpImpl::DoInferShape(Operator& op, 
                                       const HTShapeList& input_shapes, 
                                       RuntimeContext& ctx) const {
  HTShape Infer = input_shapes.at(0);
  Infer.emplace_back(num_classes());
  return {Infer};
}

Tensor MakeOnehotOp(Tensor input, size_t num_classes, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<OnehotOpImpl>(num_classes),
        {std::move(input)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu

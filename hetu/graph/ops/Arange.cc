#include "hetu/graph/ops/Arange.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ArangeOpImpl::DoCompute(Operator& op, 
                            const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Arange,
                                  start(), step(), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList ArangeOpImpl::DoGradient(Operator& op, 
                                    const TensorList& grad_outputs) const {
  return {};
}

HTShapeList ArangeOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  int64_t length = (end() - start()) / step();
  return {{length}};
}

Tensor MakeArangeOp(double start, double end, double step,
                    OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<ArangeOpImpl>(start, end, step),
           {},
           std::move(op_meta.set_is_deduce_states(false)))->output(0);
}

} // namespace graph
} // namespace hetu

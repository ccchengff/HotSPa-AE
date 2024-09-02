#include "hetu/autograd/ops/Triu.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void TriuTrilOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::TriuTril,
                                  inputs.at(0), outputs.at(0), lower(), diagonal(), stream());
}

TensorList TriuTrilOpDef::DoGradient(const TensorList& grad_outputs) {
  return {TriuTrilOp(grad_outputs.at(0), lower(), diagonal(),
                     grad_op_meta().set_name(grad_name()))->output(0)};
}

void TriuTrilOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList TriuTrilOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}


} // namespace autograd
} // namespace hetu

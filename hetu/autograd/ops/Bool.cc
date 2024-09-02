#include "hetu/autograd/ops/Bool.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BoolOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Bool,
                                  inputs.at(0), outputs.at(0), stream());
}

TensorList BoolOpDef::DoGradient(const TensorList& grad_outputs) {
  return {Tensor()};
}

void BoolOpDef::DoInferMeta() {
  auto outmeta = _inputs[0]->meta();
  AddOutput(outmeta.set_dtype(DataType::BOOL));
}

HTShapeList BoolOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

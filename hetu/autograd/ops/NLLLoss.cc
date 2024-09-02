#include "hetu/autograd/ops/NLLLoss.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace autograd {

using NLLOpDef = NLLLossOpDef;
using NLLGradOpDef = NLLLossGradientOpDef;

void NLLOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  NDArray unreduced =
    reduction() == kNONE ? outputs.at(0) : NDArray::empty(inputs.at(1)->shape(), 
                                           inputs.at(0)->device(), inputs.at(0)->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::NLLLoss, inputs.at(0),
                                  inputs.at(1), unreduced, stream());
  if (reduction() != kNONE) {
    NDArray::reduce(unreduced, reduction(), HTAxes(), false, stream_index(),
                    outputs.at(0));
  }
}

TensorList NLLOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = NLLLossGradientOp(
                      _inputs[0], _inputs[1], grad_outputs.at(0), reduction(),
                      grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input, Tensor()};
}

void NLLOpDef::DoInferMeta() {
  HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  if (_reduction != kNONE)
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape({1}).set_device(_inputs[0]->device()));
  else
    AddOutput(_inputs[1]->meta());
}

HTShapeList NLLOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else 
    return {input_shapes.at(1)};
}

void NLLGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      placement().type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      1.0 / broadcasted->numel(), broadcasted, HTAxes(), stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), stream());
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::NLLLossGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), stream());
}

void NLLGradOpDef::DoInferMeta() {
  HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  AddOutput(_inputs[0]->meta());
}

HTShapeList NLLGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

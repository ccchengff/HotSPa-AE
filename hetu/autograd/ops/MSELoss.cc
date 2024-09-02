#include "hetu/autograd/ops/MSELoss.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace autograd {

using MSEOpDef = MSELossOpDef;
using MSEGradOpDef = MSELossGradientOpDef;

void MSEOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  NDArray unreduced =
    reduction() == kNONE ? outputs.at(0) : NDArray::empty_like(inputs.at(0));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::MSELoss, inputs.at(0),
                                  inputs.at(1), unreduced, stream());
  if (reduction() != kNONE) {
    NDArray::reduce(unreduced, reduction(), HTAxes(), false, stream_index(),
                    outputs.at(0));
  }
}

TensorList MSEOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = MSELossGradientOp(
                      _inputs[0], _inputs[1], grad_outputs.at(0), reduction(),
                      grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input, Tensor()};
}

void MSEOpDef::DoInferMeta() {
  HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  if (_reduction != kNONE)
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape({1}).set_device(_inputs[0]->device()));
  else
    AddOutput(_inputs[0]->meta());
}

HTShapeList MSEOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else 
    return {input_shapes.at(0)};
}

void MSEGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      placement().type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      1.0f / broadcasted->numel(), broadcasted, HTAxes(), stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), stream());
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::MSELossGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), stream());
}

void MSEGradOpDef::DoInferMeta() {
  HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  AddOutput(_inputs[0]->meta());
}

HTShapeList MSEGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

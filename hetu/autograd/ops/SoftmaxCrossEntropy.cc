#include "hetu/autograd/ops/SoftmaxCrossEntropy.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace autograd {

using SCEOpDef = SoftmaxCrossEntropyOpDef;
using SCEGradOpDef = SoftmaxCrossEntropyGradientOpDef;

void SCEOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  HTShape output_shape = HTShape(inputs.at(0)->shape().begin(), inputs.at(0)->shape().end() - 1);
  NDArray unreduced =
    reduction() == kNONE ? outputs.at(0) : NDArray::empty(output_shape, 
                                           inputs.at(0)->device(), inputs.at(0)->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::SoftmaxCrossEntropy, inputs.at(0),
                                  inputs.at(1), unreduced, stream());
  if (reduction() != kNONE) {
    NDArray::reduce(unreduced, reduction(), HTAxes(), false, stream_index(),
                    outputs.at(0));
  }
}

TensorList SCEOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = SoftmaxCrossEntropyGradientOp(
                      _inputs[0], _inputs[1], grad_outputs.at(0), reduction(),
                      grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input, Tensor()};
}

void SCEOpDef::DoInferMeta() {
  HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  HTShape output_shape = {};
  for (size_t i = 0; i < _inputs[0]->ndim() - 1; ++i) {
    output_shape.emplace_back(_inputs[0]->shape(i));
  }
  NDArrayMeta out_meta = _inputs[0]->meta();
  if (_reduction != kNONE)
    out_meta.set_shape({1});
  else
    out_meta.set_shape(output_shape);
  AddOutput(out_meta.set_device(_inputs[0]->device()));
}

HTShapeList SCEOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else {
    HTShape output_shape = {};
    for (size_t i = 0; i < input_shapes.at(0).size() - 1; ++i) {
      output_shape.emplace_back(input_shapes.at(0)[i]);
    }
    return {output_shape};
  }
}

void SCEOpDef::DoDeduceStates() {
  DistributedStates ds_preds = _inputs[0]->get_distributed_states();
  DistributedStates ds_labels = _inputs[1]->get_distributed_states();
  int ndim = _inputs[0]->ndim();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid())
    << "SoftmaxCrossEntropyOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels))
    << "Distributed states among preds and labels should be equal!";
  HT_ASSERT(ds_preds.check_max_dim(ndim - 1)) // cannot split in last dimension
    << "Input tensor can only support split in dimension < " << ndim - 1;
  _outputs[0]->set_distributed_states(ds_preds);
}

void SCEGradOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HTShape output_shape = HTShape(inputs.at(0)->shape().begin(), inputs.at(0)->shape().end() - 1);
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty(output_shape, 
                                           inputs.at(0)->device(), inputs.at(0)->dtype());
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
    placement().type(), type(), hetu::impl::SoftmaxCrossEntropyGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), stream());
}

void SCEGradOpDef::DoInferMeta() {
  HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  AddOutput(_inputs[0]->meta());
}

HTShapeList SCEGradOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

void SCEGradOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[0]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/Gather.h"
#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void GatherOpDef::DoCompute(const NDArrayList& inputs,
                            NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Gather, inputs.at(0),
                                  inputs.at(1), outputs.at(0), get_dim(), stream());
}

TensorList GatherOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    GatherGradientOp(grad_outputs.at(0), get_dim(), _inputs[1], _inputs[0],
                     grad_op_meta().set_name(grad_name()))->output(0);
  return {grad_input, Tensor()};
}

void GatherOpDef::DoInferMeta() {
  HTShape shape;
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(_inputs[1]->shape()).set_device(_inputs[0]->device()));
}

HTShapeList
GatherOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_ASSERT(input_shapes[0].size() == input_shapes[1].size());
  int64_t len = input_shapes[0].size();
  for (int64_t i = 0; i < len; ++i) {
    if (i != get_dim())
      HT_ASSERT(input_shapes[0][i] == input_shapes[1][i]);
  }
  return {input_shapes.at(1)};
}

void GatherGradientOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs,
                                    RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::GatherGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), get_dim(), stream());
}

void GatherGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[2]->meta());
}

HTShapeList
GatherGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(2)};
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/AsStrided.h"
#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void AsStridedOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                               hetu::impl::AsStrided, inputs.at(0),
                               outputs.at(0), get_stride(), stream());
}

TensorList AsStridedOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    AsStridedGradientOp(grad_outputs.at(0), _inputs[0], get_stride(),
                        grad_op_meta().set_name(grad_name()))->output(0);
  return {grad_input};
}

void AsStridedOpDef::DoInferMeta() {
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(outshape()).set_device(_inputs[0]->device()));
}

HTShapeList
AsStridedOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {outshape()};
}

void AsStridedGradientOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::AsStridedGradient,
    inputs.at(0), outputs.at(0), get_stride(), stream());
}

void AsStridedGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
}

HTShapeList
AsStridedGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes[1]};
}

} // namespace autograd
} // namespace hetu

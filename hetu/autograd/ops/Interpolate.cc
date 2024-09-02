#include "hetu/autograd/ops/Interpolate.h"
#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void InterpolateOpDef::DoCompute(const NDArrayList& inputs,
                                 NDArrayList& outputs,
                                 RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Interpolate, inputs.at(0),
                                  outputs.at(0), align_corners(), stream());
}

TensorList InterpolateOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    InterpolateGradientOp(grad_outputs.at(0), _inputs[0], align_corners(), scale_factor(),
                          grad_op_meta().set_name(grad_name()))->output(0);
  return {grad_input, Tensor()};
}

void InterpolateOpDef::DoInferMeta() {
  HTShape output = _inputs[0]->shape();
  if (out_shape().size() == 2) {
    output[2] = out_shape()[0];
    output[3] = out_shape()[1];
  }
  else {
    HT_ASSERT(scale_factor() > 0);
    output[2] = output[2] * scale_factor();
    output[3] = output[3] * scale_factor();
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output).set_device(_inputs[0]->device()));
}

HTShapeList
InterpolateOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output = input_shapes[0];
  if (out_shape().size() == 2) {
    output[2] = out_shape()[0];
    output[3] = out_shape()[1];
  }
  else {
    HT_ASSERT(scale_factor() > 0);
    output[2] = output[2] * scale_factor();
    output[3] = output[3] * scale_factor();
  }
  return {output};
}

void InterpolateGradientOpDef::DoCompute(const NDArrayList& inputs,
                                         NDArrayList& outputs,
                                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::InterpolateGradient,
    inputs.at(0), outputs.at(0), align_corners(), stream());
}

void InterpolateGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
}

HTShapeList
InterpolateGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(1)};
}

} // namespace autograd
} // namespace hetu

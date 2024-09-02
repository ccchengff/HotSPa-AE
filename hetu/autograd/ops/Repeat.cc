#include "hetu/autograd/ops/Repeat.h"
#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void RepeatOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Repeat, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList RepeatOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    RepeatGradientOp(grad_outputs.at(0), _inputs[0],
                     grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input};
}

void RepeatOpDef::DoInferMeta() {
  HTShape output_shape = repeats();
  HT_ASSERT(output_shape.size() >= _inputs[0]->ndim())
  << output_shape << " has dim " << output_shape.size()
  << ", but " << _inputs[0]->shape() << " has dim " << _inputs[0]->ndim();
  for (size_t i = 0; i < _inputs[0]->ndim(); ++i) {
    if (_inputs[0]->shape(i) > 0)
    output_shape[i + output_shape.size() - _inputs[0]->ndim()] *= _inputs[0]->shape(i); 
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output_shape).set_device(_inputs[0]->device()));
}

HTShapeList
RepeatOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape = repeats();
  HT_ASSERT(output_shape.size() >= input_shapes[0].size());
  for (size_t i = 0; i < input_shapes[0].size(); ++i) {
    output_shape[i + output_shape.size() - input_shapes[0].size()] *= input_shapes[0][i]; 
  }
  return {output_shape};
}

void RepeatGradientOpDef::DoCompute(const NDArrayList& inputs,
                                             NDArrayList& outputs,
                                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::RepeatGradient,
    inputs.at(0), outputs.at(0), stream());
}

void RepeatGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
}

HTShapeList
RepeatGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes[1]};
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/LayerNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void LayerNormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                               hetu::impl::LayerNorm, inputs.at(0),
                               inputs.at(1), inputs.at(2), outputs.at(1), outputs.at(2),
                               outputs.at(0), normalized_shape().size(), get_eps(), stream());
}

TensorList LayerNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = LayerNormGradientOp(grad_outputs.at(0), _inputs[0],
                                        _inputs[1], _outputs[1], _outputs[2], 
                                        normalized_shape(), get_eps(), g_op_meta);
  return {grad_input->output(0), grad_input->output(1), grad_input->output(2)};
}

void LayerNormOpDef::DoInferMeta() {
  size_t dim = normalized_shape().size();
  HTShape output_shape = _inputs[0]->shape();
  for (size_t i = 0; i < dim; ++i) {
    HT_ASSERT(normalized_shape()[dim - 1 - i] == _inputs[0]->shape(_inputs[0]->ndim() - 1 - i))
    << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
    << normalized_shape() << " and input shape:" << _inputs[0]->shape();
    output_shape[_inputs[0]->ndim() - 1 - i] = 1;
  }
  AddOutput(_inputs[0]->meta());
  // TODO: scale and bias shape should be normalized_shape
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device()).set_dtype(_inputs[0]->dtype()).set_shape(output_shape));
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device()).set_dtype(_inputs[0]->dtype()).set_shape(output_shape));
}

HTShapeList LayerNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  size_t dim = normalized_shape().size();
  HTShape output_shape = input_shapes.at(0);
  for (size_t i = 0; i < dim; ++i) {
    HT_ASSERT(normalized_shape()[dim - 1 - i] == input_shapes.at(0)[input_shapes.at(0).size() - 1 - i])
    << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
    << normalized_shape() << " and input shape:" << _inputs[0]->shape();
    output_shape[input_shapes.at(0).size() - 1 - i] = 1;
  }
  // TODO: scale and bias shape should be normalized_shape
  return {input_shapes.at(0), output_shape, output_shape};
}

void LayerNormOpDef::DoDeduceStates() {
  size_t dim = normalized_shape().size();
  HTShape local_shape = _inputs[0]->shape();
  int max_dim = local_shape.size() - dim;
  auto ds_input = _inputs[0]->get_distributed_states();
  auto ds_scale = _inputs[1]->get_distributed_states();
  auto ds_bias = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_scale.is_valid() && ds_bias.is_valid()
            && ds_input.get_device_num() == ds_scale.get_device_num()
            && ds_scale.get_device_num() == ds_bias.get_device_num()) 
    << "LayerNormOpDef: input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_scale.get_dim(-2) == 1 
            && ds_bias.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(max_dim) && 
            ds_scale.check_max_dim(max_dim) &&
            ds_bias.check_max_dim(max_dim))
    << "LayerNormOp only support input, scale and bias split in dimension < " << max_dim;
  // scale and bias shape should be normalized_shape, so keep duplicate
  HT_ASSERT(ds_scale.check_pure_duplicate() && ds_bias.check_pure_duplicate())
    << "Scale and bias should be duplicate!";
  _outputs[0]->set_distributed_states(ds_input);
  _outputs[1]->set_distributed_states(ds_scale);
  _outputs[2]->set_distributed_states(ds_bias);
}

void LayerNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::LayerNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), inputs.at(3), inputs.at(4), normalized_shape().size(),
    get_eps(), stream());
}

void LayerNormGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
  AddOutput(_inputs[2]->meta());
  AddOutput(_inputs[2]->meta());
}

HTShapeList
LayerNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1), input_shapes.at(2), input_shapes.at(2)};
}

void LayerNormGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[1]->get_distributed_states());
  _outputs[1]->set_distributed_states(_inputs[2]->get_distributed_states());
  _outputs[2]->set_distributed_states(_inputs[2]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

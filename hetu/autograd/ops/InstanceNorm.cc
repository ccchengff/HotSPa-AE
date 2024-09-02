#include "hetu/autograd/ops/InstanceNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void InstanceNormOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::InstanceNorm, inputs.at(0),
    outputs.at(1), outputs.at(2), outputs.at(0), get_eps(), stream());
}

TensorList InstanceNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = InstanceNormGradientOp(grad_outputs.at(0), _inputs[0], _outputs[1], _outputs[2], get_eps(),
                                           grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input};
}

void InstanceNormOpDef::DoInferMeta() {
  HTShape local_shape = _inputs[0]->shape();
  HT_ASSERT(local_shape.size() == 4);
  local_shape[3] = 1;
  local_shape[2] = 1;
  // HT_ASSERT(save_mean->shape() == local_shape);
  // HT_ASSERT(save_var->shape() == local_shape);
  AddOutput(_inputs[0]->meta());
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(local_shape).set_device(_inputs[0]->device()));
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(local_shape).set_device(_inputs[0]->device())); 
}

HTShapeList InstanceNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape local_shape = input_shapes.at(0);
  local_shape[3] = 1;
  local_shape[2] = 1;
  return {input_shapes.at(0), local_shape, local_shape};
}

void InstanceNormOpDef::DoDeduceStates() {
  auto ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << "InstanceNormOpDef: input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(2))
    << "InstanceNormOp only support split dimensions N&C in [N, C, H, W]!";  
  _outputs[0]->set_distributed_states(ds_input);
}

void InstanceNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::InstanceNormGradient, inputs.at(0),
    inputs.at(1), outputs.at(0), const_cast<NDArray&>(inputs.at(2)),
    const_cast<NDArray&>(inputs.at(3)), get_eps(), stream());
}

void InstanceNormGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList
InstanceNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void InstanceNormGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[1]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/BatchNorm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BatchNormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  // TODO: Convert these states to VariableOps
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, const_cast<NDArray&>(inputs.at(3)), 0,
                                  stream());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ArraySet, const_cast<NDArray&>(inputs.at(4)), 1, stream());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::BatchNorm, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_momentum(), get_eps(),
    const_cast<NDArray&>(inputs.at(3)), const_cast<NDArray&>(inputs.at(4)), 
    outputs.at(1), outputs.at(2), stream());
}

TensorList BatchNormOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad = BatchNormGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                  _outputs[1], _outputs[2], get_eps(), g_op_meta);                         
  return {grad->output(0), grad->output(1), grad->output(2), Tensor(), Tensor()};
}

void BatchNormOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
  int64_t channels = _inputs[0]->shape(1);
  HTShape shape = {channels};
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device())
                         .set_dtype(_inputs[0]->dtype()).set_shape(shape));
  AddOutput(NDArrayMeta().set_device(_inputs[0]->device())
                         .set_dtype(_inputs[0]->dtype()).set_shape(shape));
}

HTShapeList BatchNormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT(input_shapes.at(0).size() == 4);
  return {input_shapes.at(0), {input_shapes.at(0)[1]}, {input_shapes.at(0)[1]}};
}

// 注: input tensor shape=[N, C, H, W], 在N, H, W维上做切分均会影响到batch norm的mean和var, 
// 导致最终结果产生差异(类比于batch和mini-batch做batchnorm的区别)
void BatchNormOpDef::DoDeduceStates() {
  auto ds_input = _inputs[0]->get_distributed_states();
  auto ds_scale = _inputs[1]->get_distributed_states();
  auto ds_bias = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << name() << ": input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(2)) // cannot split in H,W dimension
    << "Input tensor can only support split in dimension N, C!";
  HT_ASSERT(ds_input.get_dim(1) == ds_scale.get_dim(0) && ds_input.get_dim(1) == ds_bias.get_dim(0))
    << "Split states for bn_scale and bn_bias should be equal to split states for input dimension C!";  
  _outputs[0]->set_distributed_states(ds_input);
}

void BatchNormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::BatchNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), get_eps(), const_cast<NDArray&>(inputs.at(3)),
    const_cast<NDArray&>(inputs.at(4)), stream());
}

void BatchNormGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
  AddOutput(_inputs[2]->meta());
  AddOutput(_inputs[2]->meta());
}

HTShapeList
BatchNormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t channels = input_shapes.at(0)[1];
  return {input_shapes.at(1), {channels}, {channels}};
}

void BatchNormGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[1]->get_distributed_states());
  _outputs[1]->set_distributed_states(_inputs[2]->get_distributed_states());
  _outputs[2]->set_distributed_states(_inputs[2]->get_distributed_states());  
}

} // namespace autograd
} // namespace hetu

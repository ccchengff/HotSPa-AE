#include "hetu/autograd/ops/Concat.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ConcatOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::Concat, inputs.at(0), inputs.at(1),
    outputs.at(0), get_axis(), stream());
}

TensorList ConcatOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_inputA =
    ConcatGradientOp(_inputs[0], grad_outputs.at(0), get_axis(), 0,
                     g_op_meta.set_name(grad_name(0)))
      ->output(0);
  auto grad_inputB =
    ConcatGradientOp(_inputs[1], grad_outputs.at(0), get_axis(), 1,
                     g_op_meta.set_name(grad_name(0)))
      ->output(0);
  return {grad_inputA, grad_inputB};
}

void ConcatOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  HTShape shape;
  if (_inputs[0]->has_shape() && _inputs[1]->has_shape()) {
    for (size_t i = 0; i < _inputs[0]->ndim(); ++i) {
      if (i != get_axis())
        HT_ASSERT(_inputs[0]->shape(i) == _inputs[1]->shape(i));
      }
    HT_ASSERT(_inputs[0]->shape(get_axis()) >= 0 && _inputs[1]->shape(get_axis()) >= 0);
    shape = _inputs[0]->shape();
    shape[get_axis()] += _inputs[1]->shape(get_axis());
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList ConcatOpDef::DoInferShape(const HTShapeList& input_shapes) { 
  HTShape shapeA = input_shapes.at(0);
  shapeA[get_axis()] += input_shapes.at(1)[get_axis()];
  return {shapeA};
}

void ConcatOpDef::DoDeduceStates() {
  DistributedStates ds_a = _inputs[0]->get_distributed_states();
  DistributedStates ds_b = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "ConcatOpDef: distributed states for input a and input b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a & b shouldn't be partial";  
  HT_ASSERT(ds_a.check_equal(ds_b)) 
    << "Distributed states for tensor a and tensor b must be equal!";
  HT_ASSERT(ds_a.get_dim(get_axis()) == 1)
    << "Concat was not allowed in splited dimension: " << get_axis();
  _outputs[0]->set_distributed_states(ds_a);
}

void ConcatGradientOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  if (placement().is_cuda()) {
    hetu::impl::ConcatGradientCuda(inputs.at(1), outputs.at(0), get_axis(),
                                   get_id(), stream());
  } else {
    hetu::impl::ConcatGradientCpu(inputs.at(1), outputs.at(0), get_axis(),
                                  get_id(), stream());
  }
}

void ConcatGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[0]->meta());
}

HTShapeList ConcatGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void ConcatGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[0]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

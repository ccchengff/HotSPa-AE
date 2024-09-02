#include "hetu/autograd/ops/Softmax.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SoftmaxOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Softmax,
                                  inputs.at(0), outputs.at(0), get_dim(), stream());
}

TensorList SoftmaxOpDef::DoGradient(const TensorList& grad_outputs) {
  return {SoftmaxGradientOp(_outputs[0], grad_outputs.at(0), get_dim(),
                            grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void SoftmaxOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList SoftmaxOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void SoftmaxOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  int ndim = _inputs[0]->ndim();
  HT_ASSERT(ds_input.is_valid()) 
    << "SoftmaxOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(ndim - 1)) // cannot split in last dimension
    << "Input tensor can only support split in dimension < " << ndim - 1;
  _outputs[0]->set_distributed_states(ds_input);        
}

void SoftmaxGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::SoftmaxGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), get_dim(), stream());
}

void SoftmaxGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
}

HTShapeList
SoftmaxGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void SoftmaxGradientOpDef::DoDeduceStates() {  
  _outputs[0]->set_distributed_states(_inputs[1]->get_distributed_states());    
}

} // namespace autograd
} // namespace hetu

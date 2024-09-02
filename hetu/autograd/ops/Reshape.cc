#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ArrayReshapeOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Reshape, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList ArrayReshapeOpDef::DoGradient(const TensorList& grad_outputs) {
  // auto& self = reinterpret_cast<ArrayReshapeOp&>(get_self());
  if (grad_outputs.at(0).is_defined() && grad_outputs.at(0)->is_tensor())
    return {ArrayReshapeGradientOp(grad_outputs.at(0), _inputs[0],
                                  grad_op_meta().set_name(grad_name()))
              ->output(0)};
  else 
    return { Tensor() };
}

void ArrayReshapeOpDef::DoInferMeta() {
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(_output_shape).set_device(_inputs[0]->device()));
  if (_inputs[0]->has_shape())
    set_input_shape(_inputs[0]->shape());
}

HTShapeList ArrayReshapeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  size_t input_size = 1;
  HTShape input_shape = input_shapes.at(0);
  size_t input_len = input_shape.size();
  for (size_t i = 0; i < input_len; ++i) {
    input_size *= input_shape[i];
  }
  // check if there exists -1 in output_shape
  int64_t idx = -1;
  size_t cnt = 0;
  size_t output_size = 1;
  HTShape output_shape = get_output_shape();
  int64_t output_len = output_shape.size();
  for (int64_t i = 0; i < output_len; ++i) {
    if (output_shape[i] == -1) {
      idx = i;
      cnt = cnt + 1;
      HT_ASSERT(cnt != 2) << "Output shape has more than one '-1' dims. ";
    }
    output_size *= output_shape[i];
  }
  // HT_LOG_INFO << input_shape << " " << output_shape;
  if (idx == -1) {
    HT_ASSERT(input_size == output_size) << "Invalid output size.";
  } else {
    output_size = output_size * (-1);
    HT_ASSERT(input_size % output_size == 0) << "Invalid output size.";
    output_shape[idx] = input_size / output_size;
  }
  set_input_shape(input_shape);
  return {output_shape};
}

void ArrayReshapeOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ArrayReshapeOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Input tensor cannot be splited in any dimension!";
  _outputs[0]->set_distributed_states(ds_input);    
}

void ArrayReshapeGradientOpDef::DoCompute(const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Reshape, inputs.at(0),
                                  outputs.at(0), stream());
}

void ArrayReshapeGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
}

HTShapeList
ArrayReshapeGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(1)};
}

void ArrayReshapeGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[1]->get_distributed_states());    
}

} // namespace autograd
} // namespace hetu

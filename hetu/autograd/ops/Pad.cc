#include "hetu/autograd/ops/Pad.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void PadOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Pad,
                                  inputs.at(0), outputs.at(0), get_paddings(),
                                  stream(), get_mode(), get_constant());
}

TensorList PadOpDef::DoGradient(const TensorList& grad_outputs) {
  return {PadGradientOp(grad_outputs.at(0), get_paddings(), get_mode(),
                        grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void PadOpDef::DoInferMeta() {
  HTShape shape;
  if (_inputs[0]->has_shape()) {
    shape = _inputs[0]->shape();
    size_t len = _paddings.size();
    for (size_t i = 0; i < 4; ++i) {
      if (i >= (4 - len / 2)) {
        shape[i] = shape[i] + _paddings[(i - (4 - len / 2)) * 2] +
          _paddings[(i - (4 - len / 2)) * 2 + 1];
      }
    }
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList PadOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape Infer = input_shapes.at(0);
  HTShape paddings = get_paddings();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] + paddings[(i - (4 - len / 2)) * 2] +
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  return {Infer};
}

void PadOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  size_t input_shape_len = _inputs[0]->shape().size();
  size_t padding_len = get_paddings().size();
  size_t max_split_dimension = input_shape_len - padding_len / 2;
  HT_ASSERT(ds_input.is_valid()) 
    << "PadOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(max_split_dimension))
    << "PadOp only support split dimension < " << max_split_dimension;
  _outputs[0]->set_distributed_states(ds_input);  
}

void PadGradientOpDef::DoCompute(const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::PadGradient, inputs.at(0),
    outputs.at(0), get_paddings(), stream(), get_mode());
}

void PadGradientOpDef::DoInferMeta() {
  HTShape shape = _inputs[0]->shape();
  size_t len = _paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      shape[i] = shape[i] - _paddings[(i - (4 - len / 2)) * 2] -
        _paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
}

HTShapeList PadGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape Infer = input_shapes.at(0);
  HTShape paddings = get_paddings();
  size_t len = paddings.size();
  for (size_t i = 0; i < 4; ++i) {
    if (i >= (4 - len / 2)) {
      Infer[i] = Infer[i] - paddings[(i - (4 - len / 2)) * 2] -
        paddings[(i - (4 - len / 2)) * 2 + 1];
    }
  }
  return {Infer};
}

} // namespace autograd
} // namespace hetu

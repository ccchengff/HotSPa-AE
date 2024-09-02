#include "hetu/autograd/ops/Slice.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SliceOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Slice,
                                  inputs.at(0), outputs.at(0),
                                  get_begin_pos().data(), stream());
}

TensorList SliceOpDef::DoGradient(const TensorList& grad_outputs) {
  return {SliceGradientOp(grad_outputs.at(0), _outputs[0], _inputs[0], get_begin_pos(),
                          get_ori_output_shape(),
                          grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void SliceOpDef::DoInferMeta() {
  HT_ASSERT(_begin_pos.size() == _output_shape.size());
  int len = _begin_pos.size();
  for (int i = 0; i < len; ++i) {
    HT_ASSERT(_begin_pos[i] >= 0);
  }
  AddOutput(
    NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(_output_shape).set_device(_inputs[0]->device()));
}

HTShapeList SliceOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape ori_shape = input_shapes.at(0);
  HT_ASSERT(ori_shape.size() == get_begin_pos().size());
  int ndim = ori_shape.size();
  HTShape ori_output_shape = get_ori_output_shape();
  HTShape output_shape = get_output_shape();
  HTShape begin_pos = get_begin_pos();
//   HT_LOG_INFO << output_shape << ori_output_shape << ori_shape;
  for (int i = 0; i < ndim; ++i) {
    if (ori_output_shape[i] == -1) {
      output_shape[i] = ori_shape[i] - begin_pos[i];
    }
    HT_ASSERT(output_shape[i] > 0);
    HT_ASSERT(begin_pos[i] + output_shape[i] <= ori_shape[i])
    << "begin_pos" << begin_pos[i] << " ,output_shape" << output_shape[i]
    << " ,ori_shape" << ori_shape[i];
  }
//   set_ori_output_shape(ori_shape);
  set_output_shape(output_shape);
  // set_grad_output_shape(ori_shape);
  return {output_shape};
}

void SliceOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "SliceOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  // HT_ASSERT(ds_input.check_pure_duplicate())
  //   << "Input tensor cannot be splited in any dimension!";
  HTShape ori_shape = _inputs[0]->shape();
  int ndim = ori_shape.size();
  HTShape ori_output_shape = get_ori_output_shape();
  HTShape begin_pos = get_begin_pos();
  for (int i = 0; i < ndim; i++) {
    if (!(begin_pos[i] == 0 && (ori_output_shape[i] == -1 || begin_pos[i] + ori_output_shape[i] == ori_shape[i]))) {
      HT_ASSERT(ds_input.get_dim(i) == 1)
        << "Slice dimension " << i << " shouldn't be splited!"; 
    }
  }
  _outputs[0]->set_distributed_states(ds_input);      
}

void SliceGradientOpDef::DoCompute(const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::SliceGradient, inputs.at(0),
    outputs.at(0), get_begin_pos().data(), stream());
}

void SliceGradientOpDef::DoInferMeta() {
  int len = get_begin_pos().size();
  for (int i = 0; i < len; ++i) {
    HT_ASSERT(_begin_pos[i] >= 0);
  }
  AddOutput(
    NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(_output_shape).set_device(_inputs[0]->device()));
}

HTShapeList SliceGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  set_output_shape(input_shapes.at(2));
  HTShape output_shape = get_output_shape();
  HTShape begin_pos = get_begin_pos();
  HT_ASSERT(output_shape.size() > 0);
  HTShape ori_shape = input_shapes.at(0);
  HT_ASSERT(ori_shape.size() == begin_pos.size());
  int ndim = ori_shape.size();
  for (int i = 0; i < ndim; ++i) {
    HT_ASSERT(begin_pos[i] + ori_shape[i] <= output_shape[i]);
  }
  set_ori_output_shape(ori_shape);
  return {output_shape};
}

void SliceGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());  
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/Broadcast.h"
#include "hetu/autograd/ops/Reduce.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BroadcastOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  if (mode() == 0) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::Broadcast, inputs.at(0),
                                    outputs.at(0), stream());
  } else {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    outputs.at(0), get_add_axes(), stream());
  }
}

TensorList BroadcastOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = BroadcastGradientOp(grad_outputs.at(0), _inputs[0], _outputs[0],
                                        get_grad_axes(), get_grad_keep_dims(),
                                        grad_op_meta().set_name(grad_name()))
                      ->output(0);
  if (mode() == 0)
    return {grad_input, Tensor()};
  else
    return {grad_input};
}

void BroadcastOpDef::DoInferMeta() {
  if (mode() == 0) {
    AddOutput(_inputs[1]->meta());
  }
  else {
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(get_shape()).set_device(_inputs[0]->device()));
  }
}

HTShapeList BroadcastOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  if (mode() == 0) {
    HTShape output_shape = input_shapes.at(1);
    outputlist = {input_shapes.at(1)};
  } else {
    HTShape input_shape = input_shapes.at(0);
    HTShape output_shape = get_shape();
    outputlist = {output_shape};
  }
  return outputlist;
}

void BroadcastGradientOpDef::DoCompute(const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ReduceSum, inputs.at(0),
    outputs.at(0), get_axes().data(), get_axes().size(), stream());
}

void BroadcastGradientOpDef::DoInferMeta() {
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(_inputs[1]->shape()));
}

HTShapeList
BroadcastGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShapeList outputlist = {};
  HTShape input_shape = input_shapes.at(1);
  HTShape output_shape = input_shapes.at(2);
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim);
  size_t diff = ndim - input_shape.size();
  HTAxes add_axes(diff);
  HTKeepDims keep_dims(diff);
  size_t len = diff + input_shape.size();
  HTShape n_input_shape(len);
  for (size_t i = 0; i < diff; ++i) {
    add_axes[i] = i;
    keep_dims[i] = false;
    n_input_shape[i] = 1;
  }
  for (size_t i = diff; i < len; ++i) {
    n_input_shape[i] = input_shape[i - diff];
  }
  for (size_t i = 0; i < ndim; ++i) {
    if (output_shape[i] == -1) {
      output_shape[i] = n_input_shape[i];
    }
    HT_ASSERT(output_shape[i] > 0);
    HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
    if (i >= diff && input_shape[i] == 1 && output_shape[i] > 1) {
      add_axes.emplace_back(i);
      keep_dims.emplace_back(true);
    }
  }
  set_axes(add_axes);
  set_keepdims(keep_dims);
  input_shape = input_shapes.at(0);
  ndim = input_shape.size();
  HTShape axes = get_axes();
  len = axes.size();
  HTKeepDims keepdims = get_keepdims();
  set_grad_shape(input_shape);
  add_axes = {};
  for (size_t i = 0; i < len; ++i) {
    if (axes[i] < 0) {
      axes[i] += ndim;
    }
    HT_ASSERT(axes[i] >= 0 && axes[i] < int64_t(ndim));
    if (keepdims[i] == true)
      input_shape[axes[i]] = 1;
    else {
      input_shape[axes[i]] = 0;
      add_axes.emplace_back(axes[i]);
    }
  }
  set_grad_axes(add_axes);
  output_shape = {};
  for (size_t i = 0; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  if (output_shape.size() == 0)
    output_shape.emplace_back(1);
  outputlist = {output_shape};
  return outputlist;
}

} // namespace autograd
} // namespace hetu

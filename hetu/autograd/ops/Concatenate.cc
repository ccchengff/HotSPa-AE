#include "hetu/autograd/ops/Concatenate.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ConcatenateOpDef::DoCompute(const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) {
if (placement().type() == DeviceType::CPU) {
    HT_DISPATCH_KERNEL_CPU_ONLY(placement().type(), type(),
                                hetu::impl::Concatenate, inputs,
                                outputs.at(0), get_axis(), stream());
  }
  else {
    int num = num_inputs();
    size_t offset = 0;
    size_t axis = get_axis();
    for (int i = 0; i < num; ++i) {
      HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(),
                                  hetu::impl::Concatenate, inputs.at(i),
                                  outputs.at(0), axis, offset, stream());
      offset += inputs.at(i)->shape(axis);
    }
  }
}

TensorList ConcatenateOpDef::DoGradient(const TensorList& grad_outputs) {
  TensorList grads = {};
  int num = num_inputs();
  int outdim = 0; 
  size_t axis = get_axis();
  auto g_op_meta = grad_op_meta();
  for (int i = 0; i < num; ++i) {
    HT_ASSERT(_inputs[i]->shape(axis) >= 0);
    auto grad_input = ConcatenateGradientOp(
      _inputs[i], _outputs[0], grad_outputs.at(0), axis, outdim, g_op_meta.set_name(grad_name(i)))
      ->output(0);
    outdim += _inputs[i]->shape(axis);
    grads.emplace_back(grad_input);
  }
  return grads;
}

void ConcatenateOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  int len = _inputs.size();
  bool flag = true;
  for (int i = 0; i < len; ++i) {
    if (!_inputs.at(i)->has_shape()) {
      flag = false;
      break;
    }
  }
  HTShape out_shape = {};
  if (flag) {
    out_shape = _inputs.at(0)->shape();
    int n_dim = out_shape.size();
    int out_dim = out_shape[get_axis()];
    int ind = 0;
    ind += 1;
    for (int i = 1; i < len; ++i) {
      HTShape shape = _inputs.at(i)->shape();
      HT_ASSERT(shape.size() == out_shape.size());
      for (int j = 0; j < n_dim; ++j) {
        if (j != (int) _axis) {
          HT_ASSERT(shape[j] == out_shape[j] || shape[j] == -1 ||
                    out_shape[j] == -1);
        } else {
          ind += 1;
          out_dim += shape[j];
        }
      }
    }
    out_shape[_axis] = out_dim;
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(out_shape).set_device(_inputs[0]->device()));
}

HTShapeList ConcatenateOpDef::DoInferShape(const HTShapeList& input_shapes) {
  int len = input_shapes.size();
  HTShape out_shape = input_shapes.at(0);
  int n_dim = out_shape.size();
  int out_dim = out_shape[get_axis()];
  int ind = 0;
  ind += 1;
  for (int i = 1; i < len; ++i) {
    HTShape shape = input_shapes.at(i);
    HT_ASSERT(shape.size() == out_shape.size());
    for (int j = 0; j < n_dim; ++j) {
      if (j != (int) get_axis()) {
        HT_ASSERT(shape[j] == out_shape[j]);
      } else {
        ind += 1;
        out_dim += shape[j];
      }
    }
  }
  out_shape[get_axis()] = out_dim;
  return {out_shape};
}

void ConcatenateOpDef::DoDeduceStates() {
  for (auto input : _inputs) {
    DistributedStates ds_input = input->get_distributed_states();
    HT_ASSERT(ds_input.get_dim(get_axis()) == 1)
      << "Concat was not allowed in splited dimension: " << get_axis();
  }
  // 直接调用默认的states copy函数做检查和赋值
  OperatorDef::DoDeduceStates();
}

void ConcatenateGradientOpDef::DoCompute(const NDArrayList& inputs,
                                         NDArrayList& outputs,
                                         RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ConcatenateGradient, inputs.at(2),
    outputs.at(0), get_axis(), get_offset(), stream());
}

void ConcatenateGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[0]->meta());
}

HTShapeList
ConcatenateGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

void ConcatenateGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[0]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

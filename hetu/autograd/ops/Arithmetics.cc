#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/impl/utils/dispatch.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace autograd {

std::pair<HTAxes, HTKeepDims> GradInfer(const HTShapeList& input_shapes) {
    HTShape output_shape = input_shapes[3];
  HTShape input_shape = input_shapes[2];
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
  return std::make_pair(add_axes, keep_dims);
}

void AddElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::add(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList AddElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_a = AddElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = AddElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  return {grad_a, grad_b};
}

void AddElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList AddElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void AddElewiseOpDef::DoDeduceStates() {
  DistributedStates ds_a = _inputs[0]->get_distributed_states();
  DistributedStates ds_b = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "AddElewiseOpDef: distributed states for input a and input b must be valid!";
  // allow sum/sub input tensor states to be partial
  // HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
  //   << "Tensor a & b shouldn't be partial";
  // local shape
  HTShape shape_a = _inputs[0]->shape();
  HTShape shape_b = _inputs[1]->shape();
  HT_LOG_INFO << _inputs[0] << ": shape = " << shape_a << "; " << _inputs[1] << ": shape = " << shape_b;

  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    _outputs[0]->set_distributed_states(ds_a);
  } else {
    _outputs[0]->set_distributed_states(ds_b);
  }
}

void AddByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::add(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList AddByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {grad_outputs.at(0)};
}

void AddByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList AddByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void SubElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::sub(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList SubElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_a = SubElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = SubElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  return {grad_a, grad_b};  
}

void SubElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList SubElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void SubElewiseOpDef::DoDeduceStates() {
  DistributedStates ds_a = _inputs[0]->get_distributed_states();
  DistributedStates ds_b = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "SubElewiseOpDef: distributed states for input a and input b must be valid!";
  // HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
  //   << "Tensor a & b shouldn't be partial";  
  // local shape
  HTShape shape_a = _inputs[0]->shape();
  HTShape shape_b = _inputs[1]->shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    _outputs[0]->set_distributed_states(ds_a);
  } else {
    _outputs[0]->set_distributed_states(ds_b);
  }
}

void SubByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::sub(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList SubByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {grad_outputs.at(0)};
}

void SubByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList SubByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void SubFromConstOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  NDArray::sub(const_value(), inputs.at(0), stream_index(), outputs.at(0));
}

TensorList SubFromConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {NegateOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void SubFromConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList SubFromConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void NegateOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  NDArray::neg(inputs.at(0), stream_index(), outputs.at(0));
}

TensorList NegateOpDef::DoGradient(const TensorList& grad_outputs) {
  return {NegateOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void NegateOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList NegateOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void MulElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::mul(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList MulElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
//   auto grad_a = MulElewiseOp(grad_outputs.at(0), _inputs[1],
//                              g_op_meta.set_name(grad_name(0)))
//                   ->output(0);
//   auto grad_b = MulElewiseOp(grad_outputs.at(0), _inputs[0],
//                              g_op_meta.set_name(grad_name(1)))
//                   ->output(0);
  auto grad_a = MulElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = MulElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
//     HT_LOG_INFO << _inputs[0]->shape() << "\n" << _inputs[1]->shape()
//    << "\n" << grad_a->shape() << "\n" << grad_b->shape();
  return {grad_a, grad_b};
}

void MulElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList MulElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void MulElewiseOpDef::DoDeduceStates() {
  DistributedStates ds_a = _inputs[0]->get_distributed_states();
  DistributedStates ds_b = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "MulElewiseOpDef: distributed states for input a and input b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a & b shouldn't be partial";  
  // local shape
  HTShape shape_a = _inputs[0]->shape();
  HTShape shape_b = _inputs[1]->shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    _outputs[0]->set_distributed_states(ds_a);
  } else {
    _outputs[0]->set_distributed_states(ds_b);
  }
}

void MulByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::mul(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList MulByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {MulByConstOp(grad_outputs.at(0), const_value(),
                       grad_op_meta().set_name(grad_name(0)))
            ->output(0)};
}

void MulByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList MulByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void DivElewiseOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::div(inputs.at(0), inputs.at(1), stream_index(), outputs.at(0));
}

TensorList DivElewiseOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
//   //1 / b
//   auto dividend_grad = ReciprocalOp(_inputs.at(1), g_op_meta)->output(0);
//   auto grad_a = MulElewiseOp(dividend_grad, grad_outputs.at(0),
//                              g_op_meta.set_name(grad_name(0)))
//                   ->output(0);
//   // - a / (b^2) = - (a / b) / b
//   auto divisor_grad =
//     NegateOp(DivElewiseOp(_outputs[0], _inputs.at(1), g_op_meta)->output(0),
//              g_op_meta)
//       ->output(0);
//   auto grad_b = MulElewiseOp(divisor_grad, grad_outputs.at(0),
//                              g_op_meta.set_name(grad_name(1)))
//                   ->output(0);
  auto grad_a = DivElewiseGradientOp(grad_outputs.at(0), _inputs[1], _inputs[0],
                                     _outputs[0], 0,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  auto grad_b = DivElewiseGradientOp(grad_outputs.at(0), _inputs[0], _inputs[1],
                                     _outputs[0], 1,
                                     g_op_meta.set_name(grad_name(0)))->output(0);
  return {grad_a, grad_b};
}

void DivElewiseOpDef::DoInferMeta() {
  HTShape shape = Broadcast(_inputs[0]->shape(), _inputs[1]->shape());
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList DivElewiseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void DivElewiseOpDef::DoDeduceStates() {
  DistributedStates ds_a = _inputs[0]->get_distributed_states();
  DistributedStates ds_b = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "DivElewiseOpDef: distributed states for input a and input b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a & b shouldn't be partial";   
  // local shape
  HTShape shape_a = _inputs[0]->shape();
  HTShape shape_b = _inputs[1]->shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // global shape
  for (int i = 0; i < size_a; i++) {
    shape_a[i] *= ds_a.get_dim(i);
  }
  for (int i = 0; i < size_b; i++) {
    shape_b[i] *= ds_b.get_dim(i);
  }
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(ds_b.get_dim(dim_b) == 1)
        << "Dimension " << dim_b << " of tensor b shouldn't be splited!";
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(ds_a.get_dim(dim_a) == 1) 
        << "Dimension " << dim_a << " of tensor a shouldn't be splited!";      
    } else {
      HT_LOG_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a >= size_b) {
    _outputs[0]->set_distributed_states(ds_a);
  } else {
    _outputs[0]->set_distributed_states(ds_b);
  }
}

void DivByConstOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::div(inputs.at(0), const_value(), stream_index(), outputs.at(0));
}

TensorList DivByConstOpDef::DoGradient(const TensorList& grad_outputs) {
  return {DivByConstOp(grad_outputs.at(0), const_value(),
                       grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void DivByConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList DivByConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void DivFromConstOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  NDArray::div(const_value(), inputs.at(0), stream_index(), outputs.at(0));
}

TensorList DivFromConstOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  // - c / (x^2) = - (c / x) / x
  auto divisor_grad =
    NegateOp(DivElewiseOp(_outputs[0], _inputs.at(1), g_op_meta)->output(0),
             g_op_meta)
      ->output(0);
  auto grad_input = MulElewiseOp(divisor_grad, grad_outputs.at(0),
                                 g_op_meta.set_name(grad_name(1)))
                      ->output(0);
  return {grad_input};
}

void DivFromConstOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList DivFromConstOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void ReciprocalOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray::reciprocal(inputs.at(0), stream_index(), outputs.at(0));
}

TensorList ReciprocalOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  // 1 / (x^2) = (1 / x) * (1 / x)
  auto ret = MulElewiseOp(_outputs.at(0), _outputs.at(0), g_op_meta)->output(0);
  ret = NegateOp(ret, g_op_meta)->output(0);
  ret = MulElewiseOp(ret, grad_outputs.at(0), g_op_meta.set_name(grad_name()))
          ->output(0);
  return {ret};
}

void ReciprocalOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ReciprocalOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void AddElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::copy(inputs.at(0), stream_index(), unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void AddElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList AddElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void AddElewiseGradientOpDef::DoDeduceStates() {
  // ds for grad_input is equal to input; TODO: special case: input is partial
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());
}

void SubElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0)
    NDArray::copy(inputs.at(0), stream_index(), unreduced);
  else 
    NDArray::neg(inputs.at(0), stream_index(), unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void SubElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList SubElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void SubElewiseGradientOpDef::DoDeduceStates() {
  // ds for grad_input is equal to input; TODO: special case: input is partial
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());  
}

void MulElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::mul(inputs.at(0), inputs.at(1), stream_index(), unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void MulElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList MulElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void MulElewiseGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());
}

void DivElewiseGradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0) {
    NDArray::div(inputs.at(0), inputs.at(1), stream_index(), unreduced);
  }
  else {
    NDArray::mul(inputs.at(0), inputs.at(3), stream_index(), unreduced);
    NDArray::div(unreduced, inputs.at(2), stream_index(), unreduced);
    NDArray::neg(unreduced, stream_index(), unreduced); 
  }
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, stream_index(),
                    outputs.at(0));
}

void DivElewiseGradientOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList DivElewiseGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  auto grad_pair = GradInfer(input_shapes);
  set_axes(grad_pair.first);
  set_keep_dims(grad_pair.second);
  return {input_shapes.at(2)};
}

void DivElewiseGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

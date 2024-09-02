#include "hetu/autograd/ops/Conv2d.h"
#include "hetu/autograd/ops/ReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

DistributedStates conv2d_deduce_states(
  std::unordered_map<int32_t, int32_t>& l2res_map, 
  std::unordered_map<int32_t, int32_t>& r2res_map, 
  DistributedStates& ds_l, DistributedStates& ds_r) {
  int32_t device_num = ds_l.get_device_num();
  // deduce states
  std::unordered_map<int32_t, int32_t> res_states;
  auto update_res_states = [&](std::unordered_map<int32_t, int32_t>& res_states, 
    DistributedStates& ds, std::unordered_map<int32_t, int32_t>& res_map) {
    for (auto d : {0, 1}) {
      auto splits = ds.get_dim(d);
      auto new_d = res_map[d];
      HT_ASSERT(res_states.find(new_d) == res_states.end() || res_states[new_d] == splits)
        << "Result states deduced from tensor input should be equal to filter";
      res_states[new_d] = splits;
    }
  };
  update_res_states(res_states, ds_l, l2res_map);
  update_res_states(res_states, ds_r, r2res_map);
  int32_t total_splits = 1;
  for (auto& state : res_states) {
    total_splits *= state.second;
  }
  res_states[-1] = device_num / total_splits;
  // deduce order
  std::vector<int32_t> lorder = ds_l.get_order();
  std::vector<int32_t> rorder = ds_r.get_order();
  auto get_new_order = [](std::unordered_map<int32_t, int32_t>& _map,
  std::vector<int32_t>& _order) -> std::vector<int32_t> {
    std::vector<int32_t> new_order;
    for (int32_t x : _order) {
      new_order.push_back(_map[x]);
    }
    return new_order;
  };
  auto get_index = [](std::vector<int32_t>& _order, int32_t val) -> int32_t {
    auto it = std::find(_order.begin(), _order.end(), val);
    HT_ASSERT(it != _order.end()) << "dimension " << val << " is not in order!";
    return it - _order.begin();
  };
  auto new_lorder = get_new_order(l2res_map, lorder);
  auto new_rorder = get_new_order(r2res_map, rorder);
  // few cases
  if (new_lorder != new_rorder) {
    new_lorder[get_index(new_lorder, 1)] = -1;
    new_rorder[get_index(new_rorder, 0)] = -1;
    HT_ASSERT(new_lorder == new_rorder) << "new_lorder is not equal to new_rorder!";
  } else if (std::find(new_lorder.begin(), new_lorder.end(), 0) != new_lorder.end()
             && ds_l.get_dim(-1) > 1) {
    int32_t ind0 = get_index(new_lorder, 0);
    int32_t ind1 = get_index(new_lorder, 1);
    if (ind0 > ind1) {
      int32_t tmp = ind0;
      ind0 = ind1;
      ind1 = tmp;
    }
    HT_ASSERT(ind0 + 1 == ind1) << "ind0 + 1 != ind1";
    new_lorder.insert(new_lorder.begin() + ind1, -1);
  }
  std::vector<int32_t> res_order(new_lorder);
  // distributed states for output
  DistributedStates ds_result({device_num, res_states, res_order});
  return ds_result;  
}

void Conv2dOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Conv2d,
                               inputs.at(0), inputs.at(1), outputs.at(0),
                               get_padding()[0], get_padding()[1],
                               get_stride()[0], get_stride()[1], stream());
}

TensorList Conv2dOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = Conv2dGradientofDataOp(
                      _inputs[1], grad_outputs.at(0), _inputs[0], get_padding(),
                      get_stride(), g_op_meta.set_name(grad_name(0)))
                      ->output(0);
  auto grad_filter =
    Conv2dGradientofFilterOp(_inputs[0], grad_outputs.at(0), _inputs[1],
                             get_padding(), get_stride(),
                             g_op_meta.set_name(grad_name(1)))
      ->output(0);
  return {grad_input, grad_filter};
}

void Conv2dOpDef::DoInferMeta() {
  HTShape shape = {-1, -1, -1, -1};
  if (_inputs[0]->has_shape() && _inputs[1]->has_shape()) {
    int64_t N = _inputs[0]->shape(0);
    int64_t H = _inputs[0]->shape(2);
    int64_t W = _inputs[0]->shape(3);
    int64_t f_O = _inputs[1]->shape(0);
    int64_t f_H = _inputs[1]->shape(2);
    int64_t f_W = _inputs[1]->shape(3);
    int64_t out_H = (H + 2 * get_padding()[0] - f_H) / get_stride()[0] + 1;
    int64_t out_W = (W + 2 * get_padding()[1] - f_W) / get_stride()[1] + 1;
    shape = {N, f_O, out_H, out_W};
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList Conv2dOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

void Conv2dOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  DistributedStates ds_filter = _inputs[1]->get_distributed_states();
  int32_t device_num = ds_input.get_device_num();
  HT_ASSERT(ds_input.is_valid() && ds_filter.is_valid() && 
            ds_input.get_device_num() == ds_filter.get_device_num()) 
    << "Conv2dOpDef: distributed states for input and filter must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1) 
    << "Tensor input and filter shouldn't be partial";
  HT_ASSERT(ds_input.check_max_dim(2) && ds_filter.check_max_dim(2))
    << "Conv2d input & filter cannot split in H & W dimension!";
  HT_ASSERT(ds_input.get_dim(1) == ds_filter.get_dim(1))
    << "Split states in dimension C for input and in dimension C_in for filter should be equal!";

  std::unordered_map<int32_t, int32_t> l2res_map = {{-1, 1}, {0, 0}, {1, -2}};
  std::unordered_map<int32_t, int32_t> r2res_map = {{-1, 0}, {0, 1}, {1, -2}};
  _outputs[0]->set_distributed_states(conv2d_deduce_states(l2res_map, r2res_map, ds_input, ds_filter));
}

void Conv2dGradientofFilterOpDef::DoCompute(const NDArrayList& inputs,
                                            NDArrayList& outputs,
                                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::Conv2dGradientofFilter,
    inputs.at(0), inputs.at(1), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], stream());
}

void Conv2dGradientofFilterOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList
Conv2dGradientofFilterOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void Conv2dGradientofFilterOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  DistributedStates ds_grad_output = _inputs[1]->get_distributed_states();  
  DistributedStates ds_filter = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_grad_output.is_valid() && ds_filter.is_valid()) 
    << "ConcatGradientOpDef: distributed states for input and grad_output and filter must be valid!";  
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_grad_output.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1) 
    << "Tensor input and grad_output and filter shouldn't be partial";

  std::unordered_map<int32_t, int32_t> l2res_map = {{-1, 0}, {0, -2}, {1, 1}};
  std::unordered_map<int32_t, int32_t> r2res_map = {{-1, 1}, {0, -2}, {1, 0}};
  auto ds_filter_grad = conv2d_deduce_states(l2res_map, r2res_map, ds_input, ds_grad_output);
  HT_ASSERT(ds_filter.check_equal(ds_filter_grad)) 
    << "Distributed states for filter_grad should be equal to filter!";
  _outputs[0]->set_distributed_states(ds_filter_grad);
}

void Conv2dGradientofDataOpDef::DoCompute(const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::Conv2dGradientofData, inputs.at(0),
    inputs.at(1), outputs.at(0), get_padding()[0], get_padding()[1],
    get_stride()[0], get_stride()[1], stream());
}

void Conv2dGradientofDataOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[2]->meta());
}

HTShapeList
Conv2dGradientofDataOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void Conv2dGradientofDataOpDef::DoDeduceStates() {
  DistributedStates ds_filter = _inputs[0]->get_distributed_states();
  DistributedStates ds_grad_output = _inputs[1]->get_distributed_states();  
  DistributedStates ds_input = _inputs[2]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_grad_output.is_valid() && ds_filter.is_valid()) 
    << "ConcatGradientOpDef: distributed states for input and grad_output and filter must be valid!";  
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_grad_output.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1) 
    << "Tensor input and grad_output and filter shouldn't be partial";

  std::unordered_map<int32_t, int32_t> l2res_map = {{-1, 0}, {0, -2}, {1, 1}};
  std::unordered_map<int32_t, int32_t> r2res_map = {{-1, 1}, {0, 0}, {1, -2}};
  auto ds_input_grad = conv2d_deduce_states(l2res_map, r2res_map, ds_filter, ds_grad_output);
  HT_ASSERT(ds_input.check_equal(ds_input_grad))
    << "Distributed states for input_grad should be equal to input!";
  _outputs[0]->set_distributed_states(ds_input_grad);
}

void Conv2dAddBiasOpDef::DoCompute(const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::Conv2dAddBias, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], stream());
}

TensorList Conv2dAddBiasOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = Conv2dGradientofDataOp(
                      _inputs[1], grad_outputs.at(0), _inputs[0], get_padding(),
                      get_stride(), g_op_meta.set_name(grad_name(0)))
                      ->output(0);
  auto grad_filter =
    Conv2dGradientofFilterOp(_inputs[0], grad_outputs.at(0), _inputs[1],
                             get_padding(), get_stride(),
                             g_op_meta.set_name(grad_name(1)))
      ->output(0);
  auto grad_bias = ReduceSumOp(grad_outputs.at(0), {0, 2, 3}, {false},
                               g_op_meta.set_name(grad_name(2)))
                     ->output(0);
  return {grad_input, grad_filter, grad_bias};
}

void Conv2dAddBiasOpDef::DoInferMeta() {
  HTShape shape = {-1, -1, -1, -1};
  if (_inputs[0]->has_shape() && _inputs[1]->has_shape()) {
    int64_t N = _inputs[0]->shape(0);
    int64_t H = _inputs[0]->shape(2);
    int64_t W = _inputs[0]->shape(3);
    int64_t f_O = _inputs[1]->shape(0);
    int64_t f_H = _inputs[1]->shape(2);
    int64_t f_W = _inputs[1]->shape(3);
    int64_t out_H = (H + 2 * get_padding()[0] - f_H) / get_stride()[0] + 1;
    int64_t out_W = (W + 2 * get_padding()[1] - f_W) / get_stride()[1] + 1;
    shape = {N, f_O, out_H, out_W};
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
}

HTShapeList Conv2dAddBiasOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

void Conv2dAddBiasOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  DistributedStates ds_filter = _inputs[1]->get_distributed_states();
  DistributedStates ds_bias = _inputs[2]->get_distributed_states();  
  HT_ASSERT(ds_input.is_valid() && ds_filter.is_valid() && ds_bias.is_valid() &&
            ds_input.get_device_num() == ds_filter.get_device_num() && 
            ds_filter.get_device_num() == ds_bias.get_device_num()) 
    << "Conv2dOpDef: distributed states for input and filter and bias must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_filter.get_dim(-2) == 1 && ds_bias.get_dim(-2) == 1) 
    << "Tensor input and filter and bias shouldn't be partial!";
  HT_ASSERT(ds_input.check_equal(ds_bias))
    << "Distributed states of bias should be equal to input!";

  _outputs[0]->set_distributed_states(ds_input);  
}

} // namespace autograd
} // namespace hetu

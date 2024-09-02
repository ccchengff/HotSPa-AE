#include "hetu/autograd/ops/Transpose.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void TransposeOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Transpose, inputs.at(0),
                                  outputs.at(0), get_perms().data(), stream());
}

TensorList TransposeOpDef::DoGradient(const TensorList& grad_outputs) {
  const HTShape& perm = get_perms();
  HTShape grad_perm = perm;
  for (size_t i = 0; i < perm.size(); ++i) {
    grad_perm[perm[i]] = i;
  }
  return {TransposeOp(grad_outputs.at(0), grad_perm,
                      grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void TransposeOpDef::DoInferMeta() {
  HTShape res_shape = {};
  if (_inputs[0]->has_shape()) {
    HTShape ori_shape = _inputs[0]->shape();
    HTShape perm = _perms;
    HT_ASSERT(perm.size() == ori_shape.size())
      << "Invalid perm size:" << _perms << ",expect:" << _inputs[0]->shape();
    int ndim = perm.size();
    HTShape vis(ndim);
    for (int i = 0; i < ndim; ++i) {
      HT_ASSERT(perm[i] < ndim);
      HT_ASSERT(vis[perm[i]] == 0);
      vis[perm[i]]++;
    }
    res_shape = ori_shape;
    for (int i = 0; i < ndim; ++i) {
      res_shape[i] = ori_shape[perm[i]];
    }
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(res_shape).set_device(_inputs[0]->device()));
}

HTShapeList TransposeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape ori_shape = input_shapes.at(0);
  HTShape perm = get_perms();
  HT_ASSERT(perm.size() == ori_shape.size());
  int ndim = perm.size();
  HTShape vis(ndim);
  for (int i = 0; i < ndim; ++i) {
    HT_ASSERT(perm[i] < ndim);
    HT_ASSERT(vis[perm[i]] == 0);
    vis[perm[i]]++;
  }
  HTShape res_shape = ori_shape;
  for (int i = 0; i < ndim; ++i) {
    res_shape[i] = ori_shape[perm[i]];
  }
  return {res_shape};
}

void TransposeOpDef::DoDeduceStates() {
  HTShape perm = get_perms();
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "TransposeOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  std::unordered_map<int32_t, int32_t> states = ds_input.get_states();
  std::vector<int32_t> order = ds_input.get_order();
  int32_t device_num = ds_input.get_device_num();
  auto get_perm_index = [&](int32_t key) -> int32_t {
    for (int i = 0; i < perm.size(); i++) {
      if (perm[i] == key) {
        return i;
      }
    }
    return -1;
  };
  // states
  std::unordered_map<int32_t, int32_t> new_states;
  for (auto& pair: states) {
    if (pair.first >= 0) {
      new_states[get_perm_index(pair.first)] = pair.second;
    } else {
      new_states[pair.first] = pair.second;
    }
  }
  // order
  std::vector<int32_t> new_order;
  for (auto o : order) {
    if (o >= 0) {
      new_order.push_back(get_perm_index(o));
    } else {
      new_order.push_back(o);
    }
  }
  _outputs[0]->set_distributed_states({device_num, new_states, new_order});     
}

} // namespace autograd
} // namespace hetu

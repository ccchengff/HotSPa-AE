#include "hetu/autograd/ops/ScalarLikeOp.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ScalarLikeOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, outputs.at(0),
                                  scalar_value(), stream());
}

TensorList ScalarLikeOpDef::DoGradient(const TensorList& grad_outputs) {
  return {Tensor()};
}

void ScalarLikeOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList ScalarLikeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void ScalarLikeOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ScalarLikeOpDef: distributed states for input must be valid!";  
  int32_t device_num = ds_input.get_device_num();
  std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1}); // partial -> duplicate
  std::unordered_map<int32_t, int32_t> res_states = ds_input.combine_states(src2dst);
  std::vector<int32_t> res_order = ds_input.combine_order(src2dst);
  Tensor& output = _outputs[0];
  output->set_distributed_states({device_num, res_states, res_order});
}

} // namespace autograd
} // namespace hetu

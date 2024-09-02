#include "hetu/autograd/ops/Conv2dBroadcast.h"
#include "hetu/autograd/ops/Conv2dReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void Conv2dReduceSumOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Conv2dReduceSum, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList Conv2dReduceSumOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = Conv2dBroadcastOp(grad_outputs.at(0), _inputs[0],
                                      grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input};
}

HTShapeList
Conv2dReduceSumOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {{input_shapes.at(0)[1]}};
}

void Conv2dReduceSumOpDef::DoDeduceStates() {
  DistributedStates ds_input = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "Conv2dReduceSumOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Tensor input shouldn't be partial";
  for (auto i = 0; i < _inputs[0]->shape().size(); i++) {
    if (i != 1) {
      HT_ASSERT(ds_input.get_dim(i) == 1)
        << "Dimension " << i << " should not be splited except dimension 1!";
    }
  }
  _outputs[0]->set_distributed_states(ds_input);  
}
} // namespace autograd
} // namespace hetu

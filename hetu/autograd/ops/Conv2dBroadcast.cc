#include "hetu/autograd/ops/Conv2dBroadcast.h"
#include "hetu/autograd/ops/Conv2dReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void Conv2dBroadcastOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Conv2dBroadcast, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList Conv2dBroadcastOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input =
    Conv2dReduceSumOp(grad_outputs.at(0), grad_op_meta().set_name(grad_name()))
      ->output(0);
  return {grad_input, Tensor()};
}

HTShapeList
Conv2dBroadcastOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(1)};
}

void Conv2dBroadcastOpDef::DoDeduceStates() {
  DistributedStates ds_output = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_output.is_valid())
    << "Conv2dBroadcastOpDef: distributed states for output tensor must be valid!";
  HT_ASSERT(ds_output.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";    
  _outputs[0]->set_distributed_states(ds_output);
}

} // namespace autograd
} // namespace hetu

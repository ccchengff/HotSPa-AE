#include "hetu/autograd/ops/Roll.h"
#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void RollOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                               hetu::impl::Roll, inputs.at(0),
                               shifts(), dims(),
                               outputs.at(0), stream());
}

TensorList RollOpDef::DoGradient(const TensorList& grad_outputs) {
  HTShape negshifts = shifts();
  for (auto &bit: negshifts) {
    bit = - bit;
  }
  auto grad_input =
    RollOp(grad_outputs.at(0), negshifts, dims(),
           grad_op_meta().set_name(grad_name()))->output(0);
  return {grad_input};
}

void RollOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList
RollOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes[0]};
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/Sigmoid.h"
#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SigmoidOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Sigmoid, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList SigmoidOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input =
    MulElewiseOp(
      _outputs[0],
      AddByConstOp(NegateOp(_outputs[0], g_op_meta)->output(0), 1, g_op_meta)
        ->output(0),
      g_op_meta)
      ->output(0);
  return {MulElewiseOp(grad_input, grad_outputs.at(0),
                       g_op_meta.set_name(grad_name(0)))
            ->output(0)};
}

void SigmoidOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList SigmoidOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

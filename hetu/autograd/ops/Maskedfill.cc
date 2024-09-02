#include "hetu/autograd/ops/Maskedfill.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/autograd/ops/ZerosLike.h"

namespace hetu {
namespace autograd {

void MaskedfillOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Maskedfill,
                                  inputs.at(0), inputs.at(1), val(),
                                  outputs.at(0), stream());
}

TensorList MaskedfillOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = MaskedfillOp(grad_outputs.at(0), _inputs[1], 0.0,
                                 g_op_meta.set_name(grad_name(0)))->output(0);

  return {grad_input, Tensor()};
}

void  MaskedfillOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList MaskedfillOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT(input_shapes.at(0).size() == input_shapes.at(1).size())
          << input_shapes.at(0) << " " << input_shapes.at(1);
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

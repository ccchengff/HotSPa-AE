#include "hetu/autograd/ops/Where.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/autograd/ops/ZerosLike.h"

namespace hetu {
namespace autograd {

void WhereOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Where,
                                  inputs.at(0), inputs.at(1), inputs.at(2),
                                  outputs.at(0), stream());
}

TensorList WhereOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto zero_grad = ZerosLikeOp(_inputs[1], g_op_meta)->output(0);
  auto grad_inputA = WhereOp(_inputs[0], grad_outputs.at(0), zero_grad,
                             g_op_meta.set_name(grad_name(1)))
                       ->output(0);
  auto grad_inputB = WhereOp(_inputs[0], zero_grad, grad_outputs.at(0),
                             g_op_meta.set_name(grad_name(2)))
                       ->output(0);

  return {Tensor(), grad_inputA, grad_inputB};
}

void  WhereOpDef::DoInferMeta() {
  AddOutput(_inputs[1]->meta());
}

HTShapeList WhereOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HT_ASSERT(input_shapes.at(0).size() == input_shapes.at(1).size() &&
            input_shapes.at(0).size() == input_shapes.at(2).size())
          << input_shapes.at(0) << " " << input_shapes.at(1) <<
          " " << input_shapes.at(2);
  return {input_shapes.at(1)};
}

} // namespace autograd
} // namespace hetu

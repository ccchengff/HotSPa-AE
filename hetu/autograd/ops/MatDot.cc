#include "hetu/autograd/ops/MatDot.h"
#include "hetu/autograd/ops/Arithmetics.h"
#include "hetu/autograd/ops/Reduce.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void MatDotOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::MatDot, inputs.at(0),
                                  inputs.at(1), outputs.at(0), stream());
}

TensorList MatDotOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_inputA = MatDotOp(grad_outputs.at(0), _inputs[1], 1,
                              g_op_meta.set_name(grad_name(0)))
                       ->output(0);
  auto grad_inputB =
    ReduceOp(MulElewiseOp(_inputs[0], grad_outputs.at(0), g_op_meta)->output(0),
             "sum", {1}, {false}, g_op_meta.set_name(grad_name(1)))
      ->output(0);
  return {grad_inputA, grad_inputB};
}

void MatDotOpDef::DoInferMeta() {
  HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
  AddOutput(_inputs[0]->meta());
}

HTShapeList MatDotOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

void MatDotOpDef::DoDeduceStates() {
  DistributedStates ds_a = _inputs[0]->get_distributed_states();
  DistributedStates ds_b = _inputs[1]->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && 
            ds_a.get_device_num() == ds_b.get_device_num()) 
    << "MatDotOpDef: distributed states for a and b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a and b shouldn't be partial";
  // TODO: support tensor parallel
  HT_ASSERT(ds_a.check_pure_duplicate() && ds_b.check_pure_duplicate())
    << "MatDot only require input tensor a and b are both pure duplicate!";
  _outputs[0]->set_distributed_states(ds_a);
}

} // namespace autograd
} // namespace hetu

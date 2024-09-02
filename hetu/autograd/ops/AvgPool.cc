#include "hetu/autograd/ops/AvgPool.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void AvgPoolOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::AvgPool,
                               inputs.at(0), get_kernel_H(), get_kernel_W(),
                               outputs.at(0), get_padding(), get_stride(),
                               stream());
}

TensorList AvgPoolOpDef::DoGradient(const TensorList& grad_outputs) {
  return {AvgPoolGradientOp(_outputs[0], grad_outputs.at(0), _inputs[0],
                            get_kernel_H(), get_kernel_W(), get_padding(),
                            get_stride(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void AvgPoolOpDef::DoInferMeta() {
  HTShape shape = {-1, -1, -1, -1};
  if (_inputs[0]->has_shape()) {
    int64_t N = _inputs[0]->shape(0);
    int64_t C = _inputs[0]->shape(1);
    int64_t H = _inputs[0]->shape(2);
    int64_t W = _inputs[0]->shape(3);
    int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
    int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
    shape = {N, C, p_H, p_W};
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList AvgPoolOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t C = input_shapes.at(0)[1];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
  int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
  return {{N, C, p_H, p_W}};
}

void AvgPoolOpDef::DoDeduceStates() {
  DistributedStates ds = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "AvgPoolOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds.get_dim(2) == 1 && ds.get_dim(3) == 1)
    << "H & W dimension shouldn't be splited, H: "
    << ds.get_dim(2) << ", W: " << ds.get_dim(3);
  _outputs[0]->set_distributed_states(ds);
}

void AvgPoolGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::AvgPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), stream());
}

void AvgPoolGradientOpDef::DoInferMeta() {
    AddOutput(_inputs[2]->meta());
}

HTShapeList
AvgPoolGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void AvgPoolGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

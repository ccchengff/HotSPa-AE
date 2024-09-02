#include "hetu/autograd/ops/MaxPool.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void MaxPoolOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::MaxPool,
                               inputs.at(0), get_kernel_H(), get_kernel_W(),
                               outputs.at(0), get_padding(), get_stride(),
                               stream());
}

TensorList MaxPoolOpDef::DoGradient(const TensorList& grad_outputs) {
  return {MaxPoolGradientOp(_outputs[0], grad_outputs.at(0), _inputs[0],
                            get_kernel_H(), get_kernel_W(), get_padding(),
                            get_stride(), grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void MaxPoolOpDef::DoInferMeta() {
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

HTShapeList MaxPoolOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t C = input_shapes.at(0)[1];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
  int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
  return {{N, C, p_H, p_W}};
}

void MaxPoolOpDef::DoDeduceStates() {
  DistributedStates ds = _inputs[0]->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "MaxPoolOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds.get_dim(2) == 1 && ds.get_dim(3) == 1)
    << "H & W dimension shouldn't be splited, H: "
    << ds.get_dim(2) << ", W: " << ds.get_dim(3);
  _outputs[0]->set_distributed_states(ds);
}

void MaxPoolGradientOpDef::DoCompute(const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::MaxPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), stream());
}

void MaxPoolGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[2]->meta());
}

HTShapeList
MaxPoolGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void MaxPoolGradientOpDef::DoDeduceStates() {
  _outputs[0]->set_distributed_states(_inputs[2]->get_distributed_states());
}

} // namespace autograd
} // namespace hetu

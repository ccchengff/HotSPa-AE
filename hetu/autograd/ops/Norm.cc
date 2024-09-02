#include "hetu/autograd/ops/Norm.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void NormOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Norm,
                               inputs.at(0), outputs.at(0), dim(), getp(), stream());
}

TensorList NormOpDef::DoGradient(const TensorList& grad_outputs) {
  return {NormGradientOp(_inputs[0], _outputs[0], grad_outputs.at(0), getp(), dim(),
                         grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

void NormOpDef::DoInferMeta() {
  HTShape outshape = _inputs[0]->shape();
  int64_t axi = dim() >= 0? dim(): dim() + outshape.size();
  if (keepdim()) 
    outshape[axi] = 1;
  else 
    outshape.erase(outshape.begin() + axi);
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(outshape).set_device(_inputs[0]->device()));
}

HTShapeList NormOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape outshape = input_shapes.at(0);
  int64_t axi = dim() >= 0 ? dim(): dim() + outshape.size();
  if (keepdim()) 
    outshape[axi] = 1;
  else 
    outshape.erase(outshape.begin() + axi);
  return {outshape};
}

void NormGradientOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::NormGradient, inputs.at(0),
                                  inputs.at(1), inputs.at(2), outputs.at(0), dim(), getp(), stream());
}

void NormGradientOpDef::DoInferMeta() {
  AddOutput(_inputs[0]->meta());
}

HTShapeList NormGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

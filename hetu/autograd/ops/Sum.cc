#include "hetu/autograd/ops/Sum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void SumOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
  int len = inputs.size();
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ArraySet, outputs.at(0), 0,
                                  stream());
  for (int i = 0; i < len; ++i) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::AddElewise, inputs.at(i),
                                    outputs.at(0), outputs.at(0), stream());
  }
}

TensorList SumOpDef::DoGradient(const TensorList& grad_outputs) {
  TensorList grad_inputs;
  grad_inputs.reserve(num_inputs());
  for (size_t i = 0; i < num_inputs(); i++)
    grad_inputs.push_back(grad_outputs.at(0));
  return grad_inputs;
}

HTShapeList SumOpDef::DoInferShape(const HTShapeList& input_shapes) {
  int len = input_shapes.size();
  HTShape output_shape = input_shapes[0];
  for (int i = 1; i < len; ++i) {
    output_shape = Broadcast(output_shape, input_shapes[i]);
  }
  return {output_shape};
}

} // namespace autograd
} // namespace hetu

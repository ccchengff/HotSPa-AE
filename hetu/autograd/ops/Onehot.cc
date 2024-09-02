#include "hetu/autograd/ops/Onehot.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void OnehotOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Onehot, inputs.at(0),
                                  num_classes(), outputs.at(0), stream());
}

TensorList OnehotOpDef::DoGradient(const TensorList& grad_outputs) {
  return {Tensor()};
}

void OnehotOpDef::DoInferMeta() {
  HTShape shape;
  if (_inputs[0]->has_shape()) {
    shape = _inputs[0]->shape();
    shape.emplace_back(num_classes());
  }
  AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape).set_device(_inputs[0]->device()));
}

HTShapeList OnehotOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape Infer = input_shapes.at(0);
  Infer.emplace_back(num_classes());
  return {Infer};
}

} // namespace autograd
} // namespace hetu

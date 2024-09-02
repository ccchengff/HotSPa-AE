#include "hetu/autograd/ops/Arange.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ArangeOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::Arange,
                                  start(), step(), outputs.at(0), stream());
}

TensorList ArangeOpDef::DoGradient(const TensorList& grad_outputs) {
  return {};
}

void ArangeOpDef::DoInferMeta() {
  int64_t length = (end() - start()) / step();
  AddOutput(NDArrayMeta().set_dtype(DataType::FLOAT64).set_shape({length}));
}

HTShapeList ArangeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  int64_t length = (end() - start()) / step();
  return {{length}};
}

} // namespace autograd
} // namespace hetu

#include "hetu/graph/ops/SwiGLU.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SwiGLUOpImpl::DoCompute(Operator& op, 
                           const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  NDArray::swiglu(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SwiGLUOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeSwiGLUGradientOp(op->input(0), grad_outputs.at(0),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList SwiGLUOpImpl::DoInferShape(Operator& op, 
                           const HTShapeList& input_shapes, 
                           RuntimeContext& ctx) const  {
  HTShape shape(input_shapes.at(0));
  shape.back() /= 2;
  return {shape};
}

void SwiGLUGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::SwigluGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}


HTShapeList SwiGLUGradientOpImpl::DoInferShape(Operator& op,
                           const HTShapeList& input_shapes,
                           RuntimeContext& ctx) const  {
  return {input_shapes.at(0)};
}

Tensor MakeSwiGLUOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<SwiGLUOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeSwiGLUGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<SwiGLUGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu

#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class SigmoidOpImpl;
class SigmoidOp;

class SigmoidOpImpl final : public UnaryOpImpl {
 public:
  SigmoidOpImpl(bool inplace)
  : UnaryOpImpl(quote(SigmoidOp), inplace) {
  }

 protected:
  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeSigmoidOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeSigmoidInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class SigmoidGradientOpImpl final : public UnaryGradientOpImpl {
 public:
  SigmoidGradientOpImpl()
  : UnaryGradientOpImpl(quote(SigmoidGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeSigmoidGradientOp(Tensor out_grad, Tensor output, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

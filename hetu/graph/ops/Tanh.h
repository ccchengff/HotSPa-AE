#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class TanhOpImpl;
class TanhOp;
class TanhGradientOpImpl;
class TanhGradientOp;

class TanhOpImpl final : public UnaryOpImpl {

 public:
  TanhOpImpl(bool inplace)
  : UnaryOpImpl(quote(TanhOp), inplace) {
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

Tensor MakeTanhOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeTanhInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class TanhGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  TanhGradientOpImpl()
  : UnaryGradientOpImpl(quote(TanhGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeTanhGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

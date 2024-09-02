#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class SqrtOpImpl;
class SqrtOp;
class ReciprocalSqrtOpImpl;
class ReciprocalSqrtOp;

class SqrtOpImpl : public UnaryOpImpl {

 public:
  SqrtOpImpl(bool inplace)
  : UnaryOpImpl(quote(SqrtOp), inplace) {
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

Tensor MakeSqrtOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeSqrtInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class ReciprocalSqrtOpImpl : public UnaryOpImpl {

 public:
  ReciprocalSqrtOpImpl(bool inplace)
  : UnaryOpImpl(quote(ReciprocalSqrtOp), inplace) {
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

Tensor MakeReciprocalSqrtOp(Tensor grad_output, OpMeta op_meta = OpMeta());

Tensor MakeReciprocalSqrtInplaceOp(Tensor grad_output, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

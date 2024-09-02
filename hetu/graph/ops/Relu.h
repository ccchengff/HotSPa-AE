#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class ReluOpImpl;
class ReluOp;
class ReluGradientOpImpl;
class ReluGradientOp;

class ReluOpImpl final : public UnaryOpImpl {
 private:
  friend class ReluOp;
  struct constrcutor_access_key {};

 public:
  ReluOpImpl(bool inplace)
  : UnaryOpImpl(quote(ReluOp), inplace) {
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

Tensor MakeReluOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeReluInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class ReluGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  ReluGradientOpImpl()
  : UnaryGradientOpImpl(quote(ReluGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeReluGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

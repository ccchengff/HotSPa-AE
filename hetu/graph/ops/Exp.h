#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class ExpOpImpl;
class ExpOp;
class ExpGradientOpImpl;
class ExpGradientOp;

class ExpOpImpl final : public UnaryOpImpl {
 private:
  friend class ExpOp;
  struct constrcutor_access_key {};

 public:
  ExpOpImpl(bool inplace)
  : UnaryOpImpl(quote(ExpOp), inplace) {
  }

 protected:
  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeExpOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeExpInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

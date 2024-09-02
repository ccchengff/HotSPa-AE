#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class RoundOpImpl;
class RoundOp;

class RoundOpImpl final : public UnaryOpImpl {
 private:
  friend class RoundOp;
  struct constrcutor_access_key {};

 public:
  RoundOpImpl(bool inplace)
  : UnaryOpImpl(quote(RoundOp), inplace) {
  }

 protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeRoundOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeRoundInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

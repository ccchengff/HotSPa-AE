#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class PowTensorAndConstOpImpl;
class PowTensorAndConstOp;

class PowTensorAndConstOpImpl final : public UnaryOpImpl {
 private:
  friend class PowTensorAndConstOp;
  struct constrcutor_access_key {};

 public:
  PowTensorAndConstOpImpl(double exponent, bool inplace)
  : UnaryOpImpl(quote(PowTensorAndConstOp), inplace), _exponent(exponent) {
  }

  inline double exponent() const {
    return _exponent;
  }

 protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  double _exponent;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const PowTensorAndConstOpImpl&>(rhs);
      return exponent() == rhs_.exponent();
    }
    return false;
  }
};

Tensor MakePowTensorAndConstOp(Tensor input, double exponent, OpMeta op_meta = OpMeta());

Tensor MakePowTensorAndConstInplaceOp(Tensor input, double exponent, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class GeluOpImpl;
class GeluOp;
class GeluGradientOpImpl;
class GeluGradientOp;

class GeluOpImpl final : public UnaryOpImpl {
 private:
  friend class GeluOp;
  struct constrcutor_access_key {};

 public:
  GeluOpImpl()
  : UnaryOpImpl(quote(GeluOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs); 
  }
};

Tensor MakeGeluOp(Tensor input, OpMeta op_meta = OpMeta());

class GeluGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  GeluGradientOpImpl()
  : UnaryGradientOpImpl(quote(GeluGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeGeluGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

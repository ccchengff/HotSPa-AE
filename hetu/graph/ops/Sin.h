#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class SinOpImpl;
class SinOp;
class CosOpImpl;
class CosOp;

class SinOpImpl final : public UnaryOpImpl {
 private:
  friend class SinOp;
  struct constrcutor_access_key {};

 public:
  SinOpImpl(bool inplace)
  : UnaryOpImpl(quote(SinOp), inplace) {
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

Tensor MakeSinOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeSinInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

class SinGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  SinGradientOpImpl()
  : UnaryGradientOpImpl(quote(SinGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeSinGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

class CosOpImpl final : public UnaryOpImpl {

 public:
  CosOpImpl(bool inplace)
  : UnaryOpImpl(quote(CosOp), inplace) {
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

Tensor MakeCosOp(Tensor input, bool inplace = false, OpMeta op_meta = OpMeta());

class CosGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  CosGradientOpImpl()
  : UnaryGradientOpImpl(quote(CosGradientOp)) {
  }

 protected:
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeCosGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

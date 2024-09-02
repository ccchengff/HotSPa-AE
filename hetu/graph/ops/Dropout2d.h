#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class Dropout2dOpImpl;
class Dropout2dOp;
class Dropout2dGradientWithRecomputationOpImpl;
class Dropout2dGradientWithRecomputationOp;

class Dropout2dOpImpl final : public UnaryOpImpl {
 public:
  Dropout2dOpImpl(double keep_prob,
                  bool recompute = false, bool inplace = false)
  : UnaryOpImpl(quote(Dropout2dOp), inplace),
    _keep_prob(keep_prob),
    _recompute(recompute || inplace) {
    // TODO: support without recomputation
    HT_ASSERT(inplace) << "Currently we require Conv2D to be in place";
  }

  double keep_prob() const {
    return _keep_prob;
  };

  bool recompute() const {
    return _recompute;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  double _keep_prob;
  bool _recompute;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Dropout2dOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob() &&
              recompute() == rhs_.recompute());
    }
    return false;
  }
};

Tensor MakeDropout2dOp(Tensor input, double keep_prob,
                       bool recompute = false, OpMeta op_meta = OpMeta());

Tensor MakeDropout2dInplaceOp(Tensor input, double keep_prob,
                              bool recompute = false, OpMeta op_meta = OpMeta());

class Dropout2dGradientWithRecomputationOpImpl final : public UnaryGradientOpImpl {
 public:
  Dropout2dGradientWithRecomputationOpImpl(OpId forward_op,
                                           double keep_prob,
                                           bool fw_inplace,
                                           OpMeta op_meta = OpMeta())
  : UnaryGradientOpImpl(quote(Dropout2dGradientWithRecomputationOp)),
    _forward_op(forward_op),
    _keep_prob(keep_prob),
    _fw_inplace(fw_inplace) {
  }

  double keep_prob() const {
    return _keep_prob;
  }

  bool fw_inplace() const {
    return _fw_inplace;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  OpId _forward_op;

  double _keep_prob;

  bool _fw_inplace;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Dropout2dGradientWithRecomputationOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob() &&
              _forward_op == rhs_._forward_op &&
              fw_inplace() == rhs_.fw_inplace());
    }
    return false;
  }
};

Tensor MakeDropout2dGradientWithRecomputationOp(Tensor grad_output,
                                                OpId forward_op, double keep_prob,
                                                bool fw_inplace,
                                                OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

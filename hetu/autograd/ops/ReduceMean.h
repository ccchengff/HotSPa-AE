#pragma once

#include "hetu/autograd/ops/Reduce.h"

namespace hetu {
namespace autograd {

class ReduceMeanOpDef;
class ReduceMeanOp;

class ReduceMeanOpDef : public ReduceOpDef {
 private:
  friend class ReduceMeanOp;
  struct constrcutor_access_key {};

 public:
  ReduceMeanOpDef(Tensor input,
                  const HTShape& axes = {}, const HTKeepDims& keepdims = {false},
                  const OpMeta& op_meta = OpMeta())
  : ReduceOpDef(quote(ReduceMeanOp), input, ReductionType::MEAN, axes, keepdims, op_meta) {}
};

class ReduceMeanOp final : public OpWrapper<ReduceMeanOpDef> {
 public:
  ReduceMeanOp() : OpWrapper<ReduceMeanOpDef>() {}
  ReduceMeanOp(Tensor input, const HTShape& axes = {},
               const HTKeepDims& keepdims = {false},
               const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceMeanOpDef>(
      make_ptr<ReduceMeanOpDef>(input, axes, keepdims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

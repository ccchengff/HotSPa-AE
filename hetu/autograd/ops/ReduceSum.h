#pragma once

#include "hetu/autograd/ops/Reduce.h"

namespace hetu {
namespace autograd {

class ReduceSumOpDef;
class ReduceSumOp;

class ReduceSumOpDef : public ReduceOpDef {
 private:
  friend class ReduceSumOp;
  struct constrcutor_access_key {};

 public:
  ReduceSumOpDef(Tensor input,
                  const HTShape& axes = {}, const HTKeepDims& keepdims = {false},
                  const OpMeta& op_meta = OpMeta())
  : ReduceOpDef(quote(ReduceSumOp), input, ReductionType::SUM, axes, keepdims, op_meta) {}
};

class ReduceSumOp final : public OpWrapper<ReduceSumOpDef> {
 public:
  ReduceSumOp() : OpWrapper<ReduceSumOpDef>() {}
  ReduceSumOp(Tensor input, const HTShape& axes = {},
               const HTKeepDims& keepdims = {false},
               const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceSumOpDef>(
      make_ptr<ReduceSumOpDef>(input, axes, keepdims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
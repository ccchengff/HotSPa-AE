#pragma once

#include "hetu/autograd/ops/ScalarLikeOp.h"

namespace hetu {
namespace autograd {

class OnesLikeOpDef;
class OnesLikeOp;

class OnesLikeOpDef : public ScalarLikeOpDef {
 private:
  friend class OnesLikeOp;
  struct constrcutor_access_key {};

 public:
  OnesLikeOpDef(const constrcutor_access_key&, Tensor input,
                const OpMeta& op_meta = OpMeta())
  : ScalarLikeOpDef(quote(OnesLikeOp), input, 1, op_meta) {}
};

class OnesLikeOp final : public OpWrapper<OnesLikeOpDef> {
 public:
  OnesLikeOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<OnesLikeOpDef>(make_ptr<OnesLikeOpDef>(
      OnesLikeOpDef::constrcutor_access_key(), input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

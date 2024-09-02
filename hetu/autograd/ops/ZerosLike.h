#pragma once

#include "hetu/autograd/ops/ScalarLikeOp.h"

namespace hetu {
namespace autograd {

class ZerosLikeOpDef;
class ZerosLikeOp;

class ZerosLikeOpDef : public ScalarLikeOpDef {
 private:
  friend class ZerosLikeOp;
  struct constrcutor_access_key {};

 public:
  ZerosLikeOpDef(Tensor input, const OpMeta& op_meta = OpMeta())
  : ScalarLikeOpDef(quote(ZerosLikeOp), input, 0, op_meta) {}
};

class ZerosLikeOp final : public OpWrapper<ZerosLikeOpDef> {
 public:
  ZerosLikeOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ZerosLikeOpDef>(make_ptr<ZerosLikeOpDef>(input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

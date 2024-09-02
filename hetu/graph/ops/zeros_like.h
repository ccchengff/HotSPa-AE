#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/ops/scalars_like.h"

namespace hetu {
namespace graph {

class ZerosLikeOpImpl final : public ScalarsLikeOpImpl {
 public:
  ZerosLikeOpImpl()
  : ScalarsLikeOpImpl(quote(ZerosLikeOp), 0) {}

 public:
  bool operator==(const OpInterface& rhs) const {
    return ScalarsLikeOpImpl::operator==(rhs);
  }
};

Tensor MakeZerosLikeOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

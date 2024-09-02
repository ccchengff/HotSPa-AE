#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakeOnesLikeOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<OnesLikeOpImpl>(), {std::move(input)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu

#include "hetu/graph/ops/sum.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

Tensor MakeSumOp(TensorList inputs, OpMeta op_meta) {
  // DataType input_type = DataType::FLOAT32;
  //
  return Graph::MakeOp(std::make_shared<SumOpImpl>(), std::move(inputs),
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu

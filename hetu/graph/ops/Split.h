#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SplitOpImpl;
class SplitOp;
class SplitGradientOpImpl;
class SplitGradientOp;

// seems deprecated
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       OpMeta op_meta = OpMeta());

// deprecated: only used in gpt inference, before symbolic shape is realized
TensorList MakeSplitOp(Tensor input, int64_t num_chunks, int64_t dim,
                       int64_t padding_axis, OpMeta op_meta = OpMeta());

// 这里只能做到在单一的dim上的切分
// 主要用于qkv.split(3)
TensorList MakeSplitOp(Tensor input, const HTShape& chunks, int64_t dim,
                       OpMeta op_meta = OpMeta());

// 可以缺省
// 只在axes部分维度上切分
// 主要用于替换exec graph中的通信算子
Tensor MakeSplitOp(Tensor input, const HTAxes& axes, const HTShape& indices,
                   const HTShape& splits, OpMeta op_meta = OpMeta());

// 不可缺省   
// 主要用于exec graph witch时通信图的建立          
Tensor MakeSplitOp(Tensor input, const HTShape& indices, const HTShape& splits, 
                   OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu

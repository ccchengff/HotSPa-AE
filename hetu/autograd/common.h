#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include <vector>
#include <functional>

namespace hetu {
namespace autograd {

// Somethong went wrong if we remove this line...
using hetu::operator<<;

class OperatorDef;
template <typename OpDef>
class OpWrapper;
using Operator = OpWrapper<OperatorDef>;
using OpId = int64_t;
using OpType = std::string;
using OpName = std::string;
using OpList = std::vector<Operator>;
using OpRefList = std::vector<std::reference_wrapper<Operator>>;
using OpIdList = std::vector<OpId>;
using Op2OpMap = std::unordered_map<OpId, Operator>;

class TensorDef;
class Tensor;
using TensorId = int64_t;
using TensorName = std::string;
using TensorList = std::vector<Tensor>;
using TensorIdList = std::vector<TensorId>;
using Tensor2TensorMap = std::unordered_map<TensorId, Tensor>;
using Tensor2TensorListMap = std::unordered_map<TensorId, TensorList>;
using Tensor2NDArrayMap = std::unordered_map<TensorId, NDArray>;
using Tensor2ShapeMap = std::unordered_map<TensorId, HTShape>;

} // namespace autograd
} // namespace hetu

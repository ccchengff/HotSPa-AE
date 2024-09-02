#pragma once

#include "hetu/graph/tensor.h"
#include "hetu/core/ndarray_meta.h"
#include <algorithm>
#include <map>

namespace hetu {
namespace graph {

inline HTShape Broadcast(const HTShape& shape_a, const HTShape& shape_b) {
  // TODO: The current implementation is incomplete. Re-write it in the future.  
  if (shape_a == shape_b)
    return shape_a;
  
  auto _check_shape = [](const HTShape& shape) {
    for (size_t i = 1; i < shape.size(); i++)
      HT_VALUE_ERROR_IF(shape[i] == -1)
        << "Currently we only support the first dimension to be unknown, got "
        << shape;
  };
  _check_shape(shape_a);
  _check_shape(shape_b);

  HT_VALUE_ERROR_IF((shape_a.empty() && !shape_b.empty()) ||
                    (!shape_a.empty() && shape_b.empty()))
    << "Cannot infer the broadcastted shape given " << shape_a << " x "
    << shape_b;
  
  HTShape a = shape_a, b = shape_b;
  std::replace(a.begin(), a.end(), static_cast<HTShape::value_type>(-1),
               std::numeric_limits<HTShape::value_type>::max());
  std::replace(b.begin(), b.end(), static_cast<HTShape::value_type>(-1),
               std::numeric_limits<HTShape::value_type>::max());
  auto ret = NDArrayMeta::Broadcast(a, b);
  HT_VALUE_ERROR_IF(ret.empty()) << "Cannot infer the broadcastted shape given "
                                 << shape_a << " x " << shape_b;
  std::replace(ret.begin(), ret.end(),
               std::numeric_limits<HTShape::value_type>::max(),
               static_cast<HTShape::value_type>(-1));
  return ret;
}

inline std::vector<DataType> _get_dtypes(const TensorList& tensors) {
  std::vector<DataType> ret(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++)
    ret[i] = tensors.at(i)->dtype();
  return ret;
}

inline bool SameDataType(const TensorList& tensors) {
  if (tensors.size() > 0) {
    auto dtype = tensors.at(0)->dtype();
    for (size_t i = 1; i < tensors.size(); i++)
      if (tensors.at(i)->dtype() != dtype)
        return false;
  }
  return true;
}

inline bool CheckAxes(const HTAxes& axes, int64_t ndim) {
  std::unordered_set<int64_t> mset;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < -ndim || axes[i] >= ndim)
      return false;
    int64_t axi = axes[i] > 0 ? axes[i] : axes[i] + ndim;
    if (mset.find(axi) != mset.end())
      return false;
    mset.insert(axi);
  }
  return true;
}

inline bool CheckShape(const Tensor& input) {
  for (size_t i = 0; i < input->ndim(); ++i) {
    if (input->shape(i) <= 0)
      return false;
  }
}

inline bool CheckDim(const Tensor& input, int64_t ndim) {
  if (int64_t(input->ndim()) == ndim)
    return true;
  else
    return false;
}

#define HT_ASSERT_TENSORS_SAME_DTYPE(tensors)                                  \
  HT_ASSERT(SameDataType(tensors))                                             \
    << "Tensors are not with the same data type: " << _get_dtypes(tensors)

#define HT_ASSERT_CHECK_AXES(axes, ndim)                                  \
  HT_ASSERT(CheckAxes(axes, ndim))                                        \
    << "Invalid axes:" << axes;

#define HT_ASSERT_VALID_SHAPE(tensor)                                  \
  HT_ASSERT(CheckShape(tensor))                                        \
    << "Invalid shape:" << tensor->shape();


#define HT_ASSERT_HAS_DIMS(tensor, dim)                                  \
  HT_ASSERT(CheckDim(tensor, dim))                                       \
    << "Incorrect Input dims:" << tensor->ndim();

#define HT_ASSERT_TENSORS_SAME_SHAPE(tensor1, tensor2)            \
  HT_ASSERT(tensor1->shape() == tensor2->shape())                 \
    << "Tensors have different shapes "                           \
    << tensor1->shape() << " and " << tensor2->shape();

#define HT_ASSERT_TENSORS_SAME_NDIM(tensor1, tensor2)            \
  HT_ASSERT(tensor1->ndim() == tensor2->mdim())                  \
    << "Tensors have different dims "                            \
    << tensor1->ndim() << " and " << tensor2->ndim();
} // namespace graph
} // namespace hetu

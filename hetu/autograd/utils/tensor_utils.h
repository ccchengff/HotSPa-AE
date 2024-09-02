#pragma once

#include "hetu/autograd/tensor.h"
#include "hetu/core/ndarray_meta.h"
#include <algorithm>

namespace hetu {
namespace autograd {

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

#define HT_ASSERT_TENSORS_SAME_DTYPE(tensors)                                  \
  HT_ASSERT(SameDataType(tensors))                                             \
    << "Tensors are not with the same data type: " << _get_dtypes(tensors)

} // namespace autograd
} // namespace hetu
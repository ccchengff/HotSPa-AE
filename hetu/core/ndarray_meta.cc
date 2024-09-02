#include "hetu/core/ndarray_meta.h"
#include <algorithm>
#include <numeric>
#include <functional>

namespace hetu {

void NDArrayMeta::view(const HTShape& view_shape) {
  int64_t infer_dim = -1;
  for (size_t i = 0; i < view_shape.size(); i++) {
    HT_ASSERT(view_shape.at(i) >= -1) << "Invalid shape: " << view_shape;
    if (view_shape.at(i) == -1) {
      HT_ASSERT(infer_dim == -1) << "Cannot infer multiple dimensions "
                                 << "given shape " << view_shape;
      infer_dim = static_cast<int64_t>(i);
    }
  }
  int64_t cur_numel = static_cast<int64_t>(numel());
  int64_t view_numel = static_cast<int64_t>(NumEl(view_shape));
  if (infer_dim != -1) {
    view_numel *= -1;
    HT_ASSERT(cur_numel % view_numel == 0)
      << "Cannot reshape " << shape << " to " << view_shape;
    shape = view_shape;
    shape[infer_dim] = cur_numel / view_numel;
  } else {
    HT_ASSERT(view_numel == cur_numel)
      << "Cannot reshape " << shape << " to " << view_shape;
    shape = view_shape;
  }
  stride = Shape2Stride(shape);
}

void NDArrayMeta::unsqueeze(int64_t dim) {
  int64_t unsqueezed_dim = static_cast<int64_t>(ndim()) + 1;
  dim = ParseAxis(dim, unsqueezed_dim);
  int64_t new_stride = dim >= ndim() ? 1 : shape[dim] * stride[dim];
  shape.insert(shape.begin() + dim, 1);
  stride.insert(stride.begin() + dim, new_stride);
}

void NDArrayMeta::squeeze() {
  HTShape squeeze_shape;
  HTShape squeeze_stride;
  auto origin_ndim = ndim();
  for (size_t i = 0; i < origin_ndim; i++) {
    if (shape[i] != 1) {
      squeeze_shape.emplace_back(shape[i]);
      squeeze_stride.emplace_back(stride[i]);
    }
  }
  shape = std::move(squeeze_shape);
  stride = std::move(squeeze_stride);
}

void NDArrayMeta::squeeze(int64_t dim) {
  HTShape squeeze_shape;
  HTStride squeeze_stride;
  int64_t num_dim = static_cast<int64_t>(ndim());
  dim = ParseAxis(dim, num_dim);
  if (shape[dim] != 1)
    return;
  shape.erase(shape.begin() + dim);
  stride.erase(stride.begin() + dim);
}

void NDArrayMeta::flatten(int64_t start_dim, int64_t end_dim) {
  int64_t num_dim = static_cast<int64_t>(ndim());
  start_dim = ParseAxis(start_dim, num_dim);
  end_dim = ParseAxis(end_dim, num_dim);
  int64_t flat_size = 1;
  for (int64_t i = start_dim; i < end_dim + 1; ++i) {
    flat_size *= shape[i];
  }
  shape.erase(shape.begin() + start_dim, shape.begin() + end_dim + 1);
  shape.insert(shape.begin() + start_dim, flat_size);
  stride = Shape2Stride(shape);
}

void NDArrayMeta::permute(const HTAxes& axes) {
  int64_t num_dim = static_cast<int64_t>(ndim());
  HTAxes parsed_axes = ParseAxes(axes, num_dim);
  HTShape vis(num_dim, 0);
  for (int i = 0; i < num_dim; ++i) {
    HT_ASSERT(axes[i] >= 0 && axes[i] < num_dim);
    HT_ASSERT(vis[axes[i]] == 0);
    vis[axes[i]]++;
  }
  HTShape permuted_shape = shape;
  HTStride permuted_stride = stride;
  for (int64_t i = 0; i < num_dim; ++i) {
    permuted_shape[i] = shape[axes[i]];
    permuted_stride[i] = stride[axes[i]];
  }
  shape = permuted_shape;
  stride = permuted_stride;
}

HTShape NDArrayMeta::Broadcast(const HTShape& shape_a, const HTShape& shape_b) {
  HTShape ret(std::max(shape_a.size(), shape_b.size()));
  auto it_a = static_cast<int32_t>(shape_a.size()) - 1;
  auto it_b = static_cast<int32_t>(shape_b.size()) - 1;
  auto it_ret = static_cast<int32_t>(ret.size()) - 1;
  while (it_a >= 0 && it_b >= 0) {
    auto a = shape_a.at(it_a--);
    auto b = shape_b.at(it_b--);
    if (a == b || a == 1 || b == 1)
      ret[it_ret--] = std::max(a, b);
    else // cannot be broadcast together
      return HTShape();
  }
  while (it_a >= 0)
    ret[it_ret--] = shape_a.at(it_a--);
  while (it_b >= 0)
    ret[it_ret--] = shape_b.at(it_b--);
  return ret;
}

HTShape NDArrayMeta::Permute(const HTShape& shape, const HTAxes& axes) {
  int64_t num_dim = static_cast<int64_t>(shape.size());
  HTAxes parsed_axes = ParseAxes(axes, num_dim);
  HTShape vis(num_dim, 0);
  for (int i = 0; i < num_dim; ++i) {
    HT_ASSERT(axes[i] >= 0 && axes[i] < num_dim);
    HT_ASSERT(vis[axes[i]] == 0);
    vis[axes[i]]++;
  }
  HTShape permuted_shape = {};
  for (int64_t i = 0; i < num_dim; ++i) {
    permuted_shape.emplace_back(shape[axes[i]]);
  }
  return permuted_shape;
}

HTShape NDArrayMeta::Reduce(const HTShape& shape, const HTAxes& axes,
                            bool keepdims) {
  int64_t num_dim = static_cast<int64_t>(shape.size());
  HTAxes parsed_axes = ParseAxes(axes, num_dim);
  HTShape reduced_shape = shape;
  if (keepdims) {
    for (auto axis : parsed_axes)
      reduced_shape[axis] = 1;
  } else if (parsed_axes.size() == num_dim) {
    return {1};
  } else {
    HTShape reduced_shape_pre = reduced_shape;
    for (auto axis : parsed_axes)
      reduced_shape_pre[axis] = 0;
    reduced_shape = {};
    for (int i = 0; i < num_dim; i++) {
      if (reduced_shape_pre[i] != 0)
        reduced_shape.emplace_back(reduced_shape_pre[i]);
    }
  }
  return reduced_shape;
}

HTShapeList NDArrayMeta::Split(const HTShape& shape, const HTShape& chunks,
                               int axis) {
  int64_t parsed_axis = ParseAxis(axis, shape.size());
  auto sum = std::accumulate(chunks.begin(), chunks.end(),
                             static_cast<HTShape::value_type>(0));
  HT_ASSERT(sum == shape.at(parsed_axis))
    << "Cannot split axis " << axis << " of shape " << shape << " into chunks "
    << chunks;
  std::vector<HTShape> ret(chunks.size());
  for (size_t i = 0; i < chunks.size(); i++) {
    ret[i] = shape;
    ret[i][parsed_axis] = chunks[i];
  }
  return ret;
}

HTShape NDArrayMeta::Concat(const HTShapeList& shapes, int64_t axis) {
  HT_ASSERT(!shapes.empty()) << "No shapes provided.";
  HTShape ret = shapes.at(0);
  size_t ndim = ret.size();
  for (size_t i = 1; i < shapes.size(); i++) {
    HT_ASSERT(shapes.at(i).size() == ndim)
      << "Cannot concat shapes with incompatible numbers of dimensions: "
      << ndim << " vs. " << shapes.at(i).size() << " (shapes: " << shapes
      << ")";
  }
  int64_t parsed_axis = ParseAxis(axis, ndim);
  for (size_t i = 1; i < shapes.size(); i++) {
    for (size_t j = 0; j < ndim; j++) {
      if (static_cast<int64_t>(j) == parsed_axis) {
        ret[j] += shapes.at(i).at(j);
      } else {
        HT_ASSERT(shapes.at(i).at(j) == ret.at(j))
          << "Cannot concat shapes with incompatible shapes on axis " << axis
          << ": " << ret.at(j) << " vs. " << shapes.at(i).at(j)
          << " (shapes: " << shapes << ")";
      }
    }
  }
  return ret;
}

int64_t NDArrayMeta::ParseAxis(int64_t axis, int64_t num_dim) {
  HT_ASSERT(axis >= -num_dim && axis < num_dim)
    << "Invalid dimension. Expected to be within "
    << "[" << -num_dim << ", " << num_dim << "). "
    << "Got " << axis << ".";
  return axis >= 0 ? axis : axis + num_dim;
}

HTAxes NDArrayMeta::ParseAxes(const HTAxes axes, int64_t num_dim) {
  HTAxes ret;
  if (axes.size() == 0) {
    ret.resize(num_dim);
    std::iota(ret.begin(), ret.end(), 0);
  } else {
    ret.reserve(axes.size());
    std::transform(
      axes.begin(), axes.end(), std::back_inserter(ret),
      [num_dim](int64_t axis) { return ParseAxis(axis, num_dim); });
    std::sort(ret.begin(), ret.end());
    ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
  }
  return ret;
}

std::ostream& operator<<(std::ostream& os, const NDArrayMeta& meta) {
  os << "{"
     << "dtype=" << meta.dtype << ", "
     << "device=" << meta.device << ", "
     << "shape=" << meta.shape << ", "
     << "stride" << meta.stride << "}";
  return os;
}

} // namespace hetu
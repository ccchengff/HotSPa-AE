#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/kernel/Reduce.cuh"

namespace hetu {
namespace impl {

template <typename acc_t, typename factor_t>
struct MeanOp {
  factor_t factor;

  inline __device__ acc_t project(acc_t val) const {
    return val * factor;
  }

  inline __device__ acc_t reduce(acc_t acc, acc_t val) const {
    return acc + val;
  }

  MeanOp(factor_t factor) : factor(factor) {}
};

template <typename spec_t, typename acc_t = spec_t, typename out_t = spec_t>
static void mean_functor(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                         int64_t num_ax, size_t reduce_num, const Stream& stream) {
  float factor = 1.0f / reduce_num;
  auto device_id = in_arr->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  launch_reduce_kernel<spec_t, out_t, acc_t>(in_arr, out_arr, axes, num_ax,
                                             MeanOp<acc_t, float> {factor}, 0., stream);
}

void ReduceMeanCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  size_t reduce_num = 1;
  for (int64_t i = 0; i < num_ax; i++) {
    reduce_num *= in_arr->shape(axes[i]);
  }
  if (out_arr->dtype() == DataType::FLOAT16) {
    mean_functor<hetu::float16, float>(in_arr, out_arr, axes, num_ax, reduce_num, stream);
  } else if (in_arr->dtype() == DataType::FLOAT16 && out_arr->dtype() == DataType::FLOAT32) {
    mean_functor<hetu::float16, float, float>(in_arr, out_arr, axes, num_ax, reduce_num, stream);
  } else if (out_arr->dtype() == DataType::BFLOAT16) {
    mean_functor<hetu::bfloat16, float>(in_arr, out_arr, axes, num_ax, reduce_num, stream);
  } else if (in_arr->dtype() == DataType::BFLOAT16 && out_arr->dtype() == DataType::FLOAT32) {
    mean_functor<hetu::bfloat16, float, float>(in_arr, out_arr, axes, num_ax, reduce_num, stream);
  } else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    out_arr->dtype(), spec_t, "ReduceMeanCuda", [&]() {
      mean_functor<spec_t>(in_arr, out_arr, axes, num_ax, reduce_num, stream);
    });
  }
}

} // namespace impl
} // namespace hetu

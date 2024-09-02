#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/kernel/Reduce.cuh"

namespace hetu {
namespace impl {

template <typename acc_t>
struct SumOp {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return a + b;
  }
};

template <typename spec_t, typename acc_t = spec_t, typename out_t = spec_t>
struct sum_functor {
  void operator()(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                  int64_t num_ax, const Stream& stream) {
    launch_reduce_kernel<spec_t, out_t, acc_t>(in_arr, out_arr, axes, num_ax,
                                               func_wrapper<out_t, acc_t>(SumOp<acc_t>()),
                                               0., stream);
  }
};

void ReduceSumCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  if (out_arr->dtype() == DataType::FLOAT16) {
    sum_functor<hetu::float16, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else if (in_arr->dtype() == DataType::FLOAT16 && out_arr->dtype() == DataType::FLOAT32) {
    sum_functor<hetu::float16, float, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else if (out_arr->dtype() == DataType::BFLOAT16) {
    sum_functor<hetu::bfloat16, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else if (in_arr->dtype() == DataType::BFLOAT16 && out_arr->dtype() == DataType::FLOAT32) {
    sum_functor<hetu::bfloat16, float, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      out_arr->dtype(), spec_t, "ReduceSumCuda", [&]() {
        sum_functor<spec_t>{}(in_arr, out_arr, axes, num_ax, stream);
      });
  }
}

} // namespace impl
} // namespace hetu
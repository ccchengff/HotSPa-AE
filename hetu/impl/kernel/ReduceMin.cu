#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/numeric_utils.h"
#include "hetu/impl/kernel/Reduce.cuh"

namespace hetu {
namespace impl {

template <typename acc_t>
struct MinOp {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (hetu::_isnan(a) || a < b) ? a : b;
  }
};

template <typename spec_t, typename acc_t = spec_t, typename out_t = spec_t>
struct min_functor {
  void operator()(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                  int64_t num_ax, const Stream& stream) {
    launch_reduce_kernel<spec_t, out_t, acc_t>(in_arr, out_arr, axes, num_ax,
                                               func_wrapper<acc_t, acc_t>(MinOp<acc_t>()),
                                               hetu::numeric_limits<acc_t>::upper_bound(),
                                               stream);
  }
};

void ReduceMinCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  if (in_arr->dtype() == DataType::FLOAT16) {
    min_functor<hetu::float16, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else if (in_arr->dtype() == DataType::BFLOAT16) {
    min_functor<hetu::bfloat16, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "ReduceMinCuda", [&]() {
          min_functor<spec_t>{}(in_arr, out_arr, axes, num_ax, stream);
      });
  }
}

} // namespace impl
} // namespace hetu

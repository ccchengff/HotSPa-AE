#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

extern void ReduceSumCuda(const NDArray&, NDArray&, const int64_t*, int64_t,
                          const Stream&);
extern void ReduceMeanCuda(const NDArray&, NDArray&, const int64_t*, int64_t,
                           const Stream&);
extern void ReduceMaxCuda(const NDArray&, NDArray&, const int64_t*, int64_t,
                          const Stream&);
extern void ReduceMinCuda(const NDArray&, NDArray&, const int64_t*, int64_t,
                          const Stream&);
extern void ReduceProdCuda(const NDArray&, NDArray&, const int64_t*, int64_t,
                          const Stream&);

void CudnnReduceCuda(const NDArray& input, NDArray& output, const HTAxes& axes,
                     ReductionType red_type, const Stream& stream) {
  // TODO: Pack them up
  switch (red_type) {
    case kSUM:
      ReduceSumCuda(input, output, axes.data(), axes.size(), stream);
      break;
    case kMEAN:
      ReduceMeanCuda(input, output, axes.data(), axes.size(), stream);
      break;
    case kMAX:
      ReduceMaxCuda(input, output, axes.data(), axes.size(), stream);
      break;
    case kMIN:
      ReduceMinCuda(input, output, axes.data(), axes.size(), stream);
      break;
    case kPROD:
      ReduceProdCuda(input, output, axes.data(), axes.size(), stream);
      break;
    case kNONE:
      HT_NOT_IMPLEMENTED << "Reduction type cannot be none";
      __builtin_unreachable();
    default:
      HT_VALUE_ERROR << "Unknown reduction type: "
                     << static_cast<int32_t>(red_type);
      __builtin_unreachable();
  }
}

void ReduceCuda(const NDArray& input, NDArray& output, const HTAxes& axes,
                ReductionType red_type, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HTAxes parsed_axes = NDArrayMeta::ParseAxes(axes, input->ndim());
  CudnnReduceCuda(input, output, parsed_axes, red_type, stream);
}
} // namespace impl
} // namespace hetu

#include "hetu/impl/cuda/CUDARand.h"

namespace hetu {
namespace impl {

template <>
void curand_gen_uniform<float>(curandGenerator_t gen, float* ptr, size_t size) {
  CURAND_CALL(curandGenerateUniform(gen, ptr, size));
}

template <>
void curand_gen_uniform<double>(curandGenerator_t gen, double* ptr,
                                size_t size) {
  CURAND_CALL(curandGenerateUniformDouble(gen, ptr, size));
}

} // namespace impl
} // namespace hetu

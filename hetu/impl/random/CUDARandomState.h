#pragma once

#include "hetu/common/macros.h"

namespace hetu {
namespace impl {

struct CUDARandomState {
  CUDARandomState(uint64_t seed_ = 0, uint64_t offset_ = 0)
  : seed(seed_), offset(offset_) {}
  uint64_t seed;
  uint64_t offset;
};

void SetCUDARandomSeed(int32_t device_index, uint64_t seed);
void SetCUDARandomSeed(uint64_t seed);
CUDARandomState GetCUDARandomState(int32_t device_index, uint64_t seed,
                                   uint64_t num_minimum_calls);

} // namespace impl
} // namespace hetu

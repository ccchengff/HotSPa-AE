#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {
CUDARandomState cuda_random_states[HT_MAX_GPUS_COMPILE_TIME];
std::mutex cuda_random_state_mutex[HT_MAX_GPUS_COMPILE_TIME];
} // namespace

void SetCUDARandomSeed(int32_t device_index, uint64_t seed) {
  if (seed == 0)
    return;
  cuda_random_states[device_index].seed = seed;
  cuda_random_states[device_index].offset = 0;
}

void SetCUDARandomSeed(uint64_t seed) {
  if (seed == 0)
    return;
  int32_t num_devices = GetCUDADeiceCount();
  for (int32_t i = 0; i < num_devices; i++)
    SetCUDARandomSeed(i, seed);
}

CUDARandomState GetCUDARandomState(int32_t device_index, uint64_t seed,
                                   uint64_t num_minimum_calls) {
  if (seed != 0) {
    // Case 1: Kernel called with a provided seed.
    // Use it for once.
    return CUDARandomState(seed);
  } else if (cuda_random_states[device_index].seed != 0) {
    // Case 2: Kernel called without a provided seed
    // while the device has been manually seeded.
    // Share the random state.
    std::lock_guard<std::mutex> lock(cuda_random_state_mutex[device_index]);
    CUDARandomState ret = cuda_random_states[device_index];
    cuda_random_states[device_index].offset += num_minimum_calls;
    return ret;
  } else {
    // Case 3: Kernel called without a provided seed
    // and the device has not been manually seeded.
    // Generate a random seed and use it for once.
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    return CUDARandomState(seed);
  }
}

} // namespace impl
} // namespace hetu

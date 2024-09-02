#include "hetu/impl/random/CPURandomState.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {
uint64_t cpu_random_seed = 0;
std::mt19937_64 cpu_random_engine;
std::mutex cpu_random_state_mutex;
} // namespace

void SetCPURandomSeed(uint64_t seed) {
  if (seed == 0)
    return;
  cpu_random_seed = seed;
  cpu_random_engine.seed(seed);
}

uint64_t GenNextRandomSeed() {
  if (cpu_random_seed != 0) {
    // Generate random seed from the seeded engine
    std::lock_guard<std::mutex> lock(cpu_random_state_mutex);
    return cpu_random_engine();
  } else {
    // Use clock as random seed
    return std::chrono::system_clock::now().time_since_epoch().count();
  }
}

} // namespace impl
} // namespace hetu

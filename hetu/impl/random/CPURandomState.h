#pragma once

#include "hetu/common/macros.h"
#include <random>

namespace hetu {
namespace impl {

void SetCPURandomSeed(uint64_t seed);
uint64_t GenNextRandomSeed();

} // namespace impl
} // namespace hetu

#include "hetu/autograd/init/initializer.h"
#include "hetu/autograd/ops/kernel_links.h"
#include <cmath>

namespace hetu {
namespace autograd {

void GeneralizedXavierInitializer::Init(NDArray& data, uint64_t seed,
                                        StreamIndex stream_id) const {
  HT_ASSERT(data->ndim() >= 2)
    << "Number of dimensions should be at least 2. Got " << data->ndim();
  size_t hw_scale = 1;
  for (size_t i = 2; i < data->ndim(); i++)
    hw_scale *= data->shape(i);
  size_t fan_in = hw_scale * data->shape(1);
  size_t fan_out = hw_scale * data->shape(0);
  double factor = 0.0; // suppress un-initialized warning
  if (mode() == "fan_in") {
    factor = fan_in;
  } else if (mode() == "fan_out") {
    factor = fan_out;
  } else if (mode() == "avg") {
    factor = (fan_in + fan_out) * 0.5;
  } else {
    HT_VALUE_ERROR << "Invalid mode: " << mode();
    __builtin_unreachable();
  }
  if (dist() == "uniform") {
    double limit = std::sqrt(gain() / factor);
    NDArray::uniform_(data, -limit, limit, seed, stream_id);
  } else if (dist() == "normal") {
    double stddev = std::sqrt(gain() / factor);
    NDArray::normal_(data, 0, stddev, seed, stream_id);
  } else {
    HT_VALUE_ERROR << "Invalid dist: " << dist();
    __builtin_unreachable();
  }
}

} // namespace autograd
} // namespace hetu

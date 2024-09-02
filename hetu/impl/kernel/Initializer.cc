#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/random/CPURandomState.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <random>

namespace hetu {
namespace impl {

template <typename spec_t>
void init_normal_cpu(spec_t* arr, size_t size, spec_t mean, spec_t stddev,
                     uint64_t seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<double> dist(mean, stddev);
  for (size_t i = 0; i < size; i++)
    arr[i] = spec_t(dist(engine));
}

template <typename spec_t>
void init_uniform_cpu(spec_t* arr, size_t size, spec_t lb, spec_t ub,
                      uint64_t seed) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<double> dist(lb, ub);
  for (size_t i = 0; i < size; i++)
    arr[i] = spec_t(dist(engine));
}

template <typename spec_t>
void init_truncated_normal_cpu(spec_t* arr, size_t size, spec_t mean,
                               spec_t stddev, spec_t lb, spec_t ub,
                               uint64_t seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<double> dist(mean, stddev);
  for (size_t i = 0; i < size; i++) {
    do {
      arr[i] = spec_t(dist(engine));
    } while (arr[i] < lb || arr[i] > ub);
  }
}

void NormalInitsCpu(NDArray& data, double mean, double stddev, uint64_t seed,
                    const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  CPUStream cpu_stream(stream);

  size_t size = data->numel();
  if (size == 0)
    return;
  if (seed == 0)
    seed = GenNextRandomSeed();
  HT_DISPATCH_FLOATING_TYPES(data->dtype(), spec_t, "NormalInitsCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [data, size, mean, stddev, seed]() {
      init_normal_cpu<spec_t>(data->data_ptr<spec_t>(), size,
                              static_cast<spec_t>(mean),
                              static_cast<spec_t>(stddev), seed);
      },"NormalInits");    
  });
  NDArray::MarkUsedBy({data}, stream);
}

void UniformInitsCpu(NDArray& data, double lb, double ub, uint64_t seed,
                     const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  HT_ASSERT(lb < ub) << "Invalid range for uniform random init: "
                     << "[" << lb << ", " << ub << ").";
  CPUStream cpu_stream(stream);

  size_t size = data->numel();
  if (size == 0)
    return;
  if (seed == 0)
    seed = GenNextRandomSeed();
  HT_DISPATCH_FLOATING_TYPES(data->dtype(), spec_t, "UniformInitCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [data, size, lb, ub, seed]() {
      init_uniform_cpu<spec_t>(data->data_ptr<spec_t>(), size,
                               static_cast<spec_t>(lb), static_cast<spec_t>(ub),
                               seed);
      },"UniformInit");   
  });
  NDArray::MarkUsedBy({data}, stream);
}

void TruncatedNormalInitsCpu(NDArray& data, double mean, double stddev,
                             double lb, double ub, uint64_t seed,
                             const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  CPUStream cpu_stream(stream);

  size_t size = data->numel();
  if (size == 0)
    return;
  if (seed == 0)
    seed = GenNextRandomSeed();
  HT_DISPATCH_FLOATING_TYPES(
    data->dtype(), spec_t, "TruncatedNormalInitsCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [data, size, mean, stddev, lb, ub, seed]() {
      init_truncated_normal_cpu<spec_t>(
        data->data_ptr<spec_t>(), size, static_cast<spec_t>(mean),
        static_cast<spec_t>(stddev), static_cast<spec_t>(lb),
        static_cast<spec_t>(ub), seed);
      },"TruncatedNormalInits");    
  });
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu

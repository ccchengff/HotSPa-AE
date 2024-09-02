#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"

namespace hetu {

void Stream::Sync() const {
  if (_device.is_cpu()) {
    hetu::impl::CPUStream(*this).Sync();
  } else if (_device.is_cuda()) {
    hetu::impl::CUDAStream(*this).Sync();
  }
}

std::ostream& operator<<(std::ostream& os, const Stream& stream) {
  os << "stream(" << stream.device()
     << ", stream_index=" << stream.stream_index() << ")";
  return os;
}

void SynchronizeAllStreams(const Device& device) {
  if (device.is_cpu()) {
    hetu::impl::SynchronizeAllCPUStreams();
  } else if (device.is_cuda()) {
    hetu::impl::SynchronizeAllCUDAStreams(device);
  } else {
    hetu::impl::SynchronizeAllCPUStreams();
    hetu::impl::SynchronizeAllCUDAStreams(device);
  }
}

} // namespace hetu

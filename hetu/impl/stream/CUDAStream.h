#pragma once

#include "hetu/core/stream.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

// Note: we do not inherit CUDAStream from Stream
// to avoid including the CUDA-related headers in core
class CUDAStream final {
 public:
  CUDAStream(const Stream& stream);

  void Sync() {
    CudaStreamSynchronize(cuda_stream());
  }

  cudaStream_t cuda_stream() const noexcept;

  // Implicit call to `cuda_stream`
  inline operator cudaStream_t() const {
    return cuda_stream();
  }

  inline DeviceIndex device_id() const noexcept {
    return _device_id;
  }

  inline StreamIndex stream_id() const noexcept {
    return _stream_id;
  }

 private:
  const DeviceIndex _device_id;
  const StreamIndex _stream_id;
};

inline CUDAStream GetCUDAStream(int32_t device_id, StreamIndex stream_id) {
  return CUDAStream(Stream(Device(kCUDA, device_id), stream_id));
}

inline CUDAStream GetCUDAComputingStream(int32_t device_id) {
  return GetCUDAStream(device_id, kComputingStream);
}

int GetCUDADeiceCount();
void SynchronizeAllCUDAStreams(const Device& device = {});

class CUDAEvent final : public Event {
 public:
  CUDAEvent(Device device, bool enable_timing = true)
  : Event(std::move(device), enable_timing) {
    HT_ASSERT(this->device().is_cuda())
      << "CUDAEvent should be used with CUDA devices. "
      << "Got " << this->device();
    hetu::cuda::CUDADeviceGuard guard(this->device().index());
    CudaEventCreate(&_event,
                    enable_timing ? cudaEventDefault : cudaEventDisableTiming);
  }

  ~CUDAEvent() {
    hetu::cuda::CUDADeviceGuard guard(device().index());
    cudaError_t err = cudaEventDestroy(_event);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      __HT_FATAL_SILENT(hetu::cuda::cuda_error)
      << "Cuda call CudaEventDestroy failed" << cudaGetErrorString(err);
    }
  }

  inline bool IsRecorded() {
    return _recorded;
  }

  inline void Record(const Stream& stream) {
    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    CudaEventRecord(_event, CUDAStream(stream));
    _recorded = true;
  }

  inline void Sync() {
    HT_ASSERT(_recorded) << "Event has not been recorded";
    CudaEventSynchronize(_event);
  }

  inline void Block(const Stream& stream) {
    HT_ASSERT(_recorded) << "Event has not been recorded";
    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    CudaStreamWaitEvent(CUDAStream(stream), _event, 0);
  }

  inline cudaError_t Query() const {
    return cudaEventQuery(_event);
  }

  inline int64_t TimeSince(const Event& event) const {
    HT_VALUE_ERROR_IF(!enable_timing() || !event.enable_timing())
      << "Cannot measure time when timing is disabled";
    const auto& e = reinterpret_cast<const CUDAEvent&>(event);
    HT_ASSERT(e._recorded && _recorded || !e._recorded && !_recorded)
      << "Only one of Start/Stop event has been recorded!";
    if (!e._recorded && !_recorded) {
      return 0;
    } else {
      float ms;
      CudaEventElapsedTime(&ms, e._event, _event);
      return static_cast<int64_t>(ms * 1000000);
    }
  }

 private:
  cudaEvent_t _event;
  bool _recorded{false};
};

} // namespace impl
} // namespace hetu

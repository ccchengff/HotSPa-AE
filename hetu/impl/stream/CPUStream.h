#pragma once

#include "hetu/core/stream.h"
#include <functional>
#include <chrono>
#include <condition_variable>
#include <future>

namespace hetu {
namespace impl {

class CPUStream final {
 public:
  CPUStream(const Stream& stream);

  std::future<void> EnqueueTask(std::function<void()> f,
                                const std::string& name = "");

  void Sync();

  inline StreamIndex stream_id() const noexcept {
    return _stream_id;
  }

 private:
  const StreamIndex _stream_id;
};

inline CPUStream GetCPUStream(StreamIndex stream_id) {
  return CPUStream(Stream(Device(kCPU), stream_id));
}

inline CPUStream GetCPUComputingStream() {
  return GetCPUStream(kComputingStream);
}

void SynchronizeAllCPUStreams();

class CPUEvent final : public Event {
 public:
  CPUEvent(bool enable_timing = true) : Event(Device(kCPU), enable_timing) {
    _record_fn = std::bind(CPUEvent::_Record, this);
    _block_fn = std::bind(CPUEvent::_Block, this);
  }

  inline bool IsRecorded() {
    return _recorded;
  }

  inline void Record(const Stream& stream) {
    _record_fn_completed = false;
    _record_future = CPUStream(stream).EnqueueTask(_record_fn, "Event_Record");
    _recorded = true;
  }

  inline void Sync() {
    HT_ASSERT(_recorded) << "Event has not been recorded";
    if (!_record_fn_completed && _record_future.valid())
      _record_future.wait();
  }

  inline void Block(const Stream& stream) {
    HT_ASSERT(_recorded) << "Event has not been recorded";
    CPUStream(stream).EnqueueTask(_block_fn, "Event_Block");
  }

  inline int64_t TimeSince(const Event& event) const {
    HT_VALUE_ERROR_IF(!enable_timing() || !event.enable_timing())
      << "Cannot measure time when timing is disabled";
    const auto& e = reinterpret_cast<const CPUEvent&>(event);
    HT_ASSERT(e._recorded && _recorded || !e._recorded && !_recorded) 
      << "Only one of Start/Stop event has been recorded!";
    if (!e._recorded && !_recorded) 
      return 0;
    else
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
        _recorded_at - e._recorded_at).count();
  }

 private:
  static void _Record(CPUEvent* const event) {
    if (event->enable_timing())
      event->_recorded_at = std::chrono::steady_clock::now();
    event->_record_fn_completed = true;
  }

  static void _Block(CPUEvent* const event) {
    event->Sync();
  }

  std::chrono::time_point<std::chrono::steady_clock> _recorded_at;
  bool _recorded{false};
  bool _record_fn_completed;
  std::future<void> _record_future;
  std::function<void()> _record_fn;
  std::function<void()> _block_fn;
};

} // namespace impl
} // namespace hetu

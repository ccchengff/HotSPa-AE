#include "hetu/impl/stream/CPUStream.h"
#include "hetu/utils/task_queue.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

static std::once_flag
  cpu_stream_task_queue_init_flags[HT_NUM_STREAMS_PER_DEVICE];
static std::vector<std::unique_ptr<TaskQueue>>
  cpu_stream_task_queues(HT_NUM_STREAMS_PER_DEVICE);

static void InitTaskQueueForCPUStream(StreamIndex stream_index) {
  HT_ASSERT(cpu_stream_task_queues[stream_index] == nullptr)
    << "CPUStream task queues must be initialized by calling "
    << "InitTaskQueueForCPUStreamOnce";
  cpu_stream_task_queues[stream_index].reset(
    new TaskQueue("CPUStream(" + std::to_string(stream_index) + ")", 1));
}

static void InitTaskQueueForCPUStreamOnce(StreamIndex stream_index) {
  std::call_once(cpu_stream_task_queue_init_flags[stream_index],
                 InitTaskQueueForCPUStream, stream_index);
}

} // namespace

CPUStream::CPUStream(const Stream& stream) : _stream_id{stream.stream_index()} {
  HT_ASSERT(stream.device().is_cpu())
    << "Initializing CPU stream "
    << "for non-host device: " << stream.device();
  HT_ASSERT(_stream_id >= kBlockingStream &&
            _stream_id < HT_NUM_STREAMS_PER_DEVICE)
    << "Invalid device stream id: " << _stream_id;
}

std::future<void> CPUStream::EnqueueTask(std::function<void()> f,
                                         const std::string& name) {
  if (_stream_id == kBlockingStream) {
    f();
    return std::future<void>();
  } else {
    InitTaskQueueForCPUStreamOnce(_stream_id);
    return cpu_stream_task_queues[_stream_id]->Enqueue(f, name);
  }
}

void CPUStream::Sync() {
  if (_stream_id == kBlockingStream ||
      cpu_stream_task_queues[_stream_id] == nullptr ||
      !cpu_stream_task_queues[_stream_id]->running())
    return;
  // Walkaround: Instead of blocking the task queues,
  // we create an event for simplicity.
  CPUEvent event;
  event.Record(Stream(Device(kCPU), _stream_id));
  event.Sync();
}

void SynchronizeAllCPUStreams() {
  for (size_t i = 0; i < cpu_stream_task_queues.size(); i++)
    CPUStream(Stream(kCPU, i)).Sync();
}

} // namespace impl
} // namespace hetu

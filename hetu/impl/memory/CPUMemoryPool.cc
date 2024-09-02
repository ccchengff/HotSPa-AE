#include "hetu/impl/memory/CPUMemoryPool.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

inline static void batch_sync_dependent_events(
  std::unordered_map<Stream, std::shared_ptr<Event>>& events) {
  for (auto& kv : events)
    kv.second->Sync();
}

} // namespace

CPUMemoryPool::CPUMemoryPool() : MemoryPool(Device(kCPU), "CPUMemPool") {
  _data_ptr_info.reserve(8192);
  _free_on_alloc_stream_fn =
    std::bind(CPUMemoryPool::_FreeOnAllocStream, this, std::placeholders::_1);
  _free_on_join_stream_fn =
    std::bind(CPUMemoryPool::_FreeOnJoinStream, this, std::placeholders::_1);
}

CPUMemoryPool::~CPUMemoryPool() {
  std::lock_guard<std::mutex> lock(_mtx);
  CPUStream(Stream(Device(kCPU), kJoinStream)).Sync();
}

DataPtr CPUMemoryPool::AllocDataSpace(size_t num_bytes, const Stream& stream) {
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, Device(kCPU), static_cast<uint64_t>(-1)};

  std::lock_guard<std::mutex> lock(_mtx);
  auto alignment = get_data_alignment();
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  
  // Currently the allocation on host memory is blocking. Remember to check for
  // the synchronization of allocation stream when freeing or waiting 
  // if the allocation becomes non-blocking.
  void* ptr;
  int err = posix_memalign(&ptr, alignment, aligned_num_bytes);
  HT_BAD_ALLOC_IF(err != 0)
    << "Failed to allocate " << aligned_num_bytes
    << " bytes of host memory. Error: " << strerror(err);
  DataPtr data_ptr{ptr, aligned_num_bytes, Device(kCPU), next_id(), true};
  _allocated += aligned_num_bytes;
  _peak_allocated = MAX(_peak_allocated, _allocated);
  _alloc_cnt++;

  // Note: The `stream` argument might be a non-CPU stream
  // (e.g., allocated for device to host copy).
  // Since we would mark the data space used by that non-CPU stream later,
  // we simply set the `alloc_stream` as join stream here.
  // During the deallocation, the non-CPU stream would be synchronized
  // before the join stream can deallocate the memory.
  Stream alloc_stream =
    stream.device().is_cpu() ? stream : Stream(Device(kCPU), kJoinStream);
  auto insertion = _data_ptr_info.emplace(
    data_ptr.id, 
    CPUDataPtrInfo(data_ptr.ptr, aligned_num_bytes, alloc_stream));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";

  return data_ptr;
}

DataPtr CPUMemoryPool::BorrowDataSpace(void* ptr, size_t num_bytes,
                                       DataPtrDeleter deleter,
                                       const Stream& stream) {
  HT_VALUE_ERROR_IF(ptr == nullptr || num_bytes == 0)
    << "Borrowing an empty storage is not allowed";
  HT_VALUE_ERROR_IF(!deleter)
    << "Deleter must not be empty when borrowing storages";
  HT_VALUE_ERROR_IF(stream.is_defined() && !stream.is_blocking())
    << "Stream must be blocking if provided";
  
  std::lock_guard<std::mutex> lock(_mtx);
  // Note: The borrowed memory must be ready, so we use blocking stream here
  DataPtr data_ptr{ptr, num_bytes, Device(kCPU), next_id()};
  Stream alloc_stream = Stream(Device(kCPU), kBlockingStream);
  auto insertion =
    _data_ptr_info.emplace(data_ptr.id,
                           CPUDataPtrInfo(data_ptr.ptr, data_ptr.size,
                                          alloc_stream, std::move(deleter)));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";
  _borrow_cnt++;

  return data_ptr;
}

void CPUMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;

  std::lock_guard<std::mutex> lock(_mtx);

  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& dependent_events = it->second.dependent_events;

  // Two cases for deallocation:
  // (1) Never used or only used by allocation stream: 
  //     --- enqueue an async task in allocation stream to free directly.
  // (2) Used by streams other than allocation stream:
  //     --- enqueue an async task in join stream to wait for the events.
  // `FreeDataSpace` would first hold the mutex and determine which case to go,
  // and then the free fn holds the mutex later.
  if (dependent_events.empty() ||
      (dependent_events.size() == 1 &&
        dependent_events.begin()->first == alloc_stream)) {
    // If we do not ignore the blocking stream, it will immediately call the 
    // `_free_on_alloc_stream_fn` and try to acquire the mutex, 
    // leading to deadlock.
    if (!alloc_stream.is_blocking()) {
      CPUStream(alloc_stream)
        .EnqueueTask(
          [this, data_ptr]() { this->_free_on_alloc_stream_fn(data_ptr); },
          "FreeOnAllocStream");
    }
  } else {
    CPUStream(Stream(Device(kCPU), kJoinStream))
      .EnqueueTask(
        [this, data_ptr]() { this->_free_on_join_stream_fn(data_ptr); },
        "FreeOnJoinStream");
  }
}

void CPUMemoryPool::_FreeOnAllocStream(CPUMemoryPool* const pool,
                                       DataPtr data_ptr) {
  std::lock_guard<std::mutex> lock(pool->_mtx);
  auto it = pool->_data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == pool->_data_ptr_info.end())
    << "Cannot find data " << data_ptr << " in from info";
  if (it->second.deleter) {
    it->second.deleter(data_ptr);
  } else {
    free(data_ptr.ptr);
  }
  pool->_allocated -= data_ptr.size;
  pool->_data_ptr_info.erase(it);
  pool->_free_cnt++;
}

void CPUMemoryPool::_FreeOnJoinStream(CPUMemoryPool* const pool,
                                      DataPtr data_ptr) {
  std::unique_lock<std::mutex> lock(pool->_mtx);
  auto it = pool->_data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == pool->_data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  // Note: Currently the allocation on host memory is blocking, 
  // so it is ok if the allocation stream is not marked.
  auto& dependent_events = it->second.dependent_events;
  // We do not need to lock the memory pool when waiting for the events
  lock.unlock();
  batch_sync_dependent_events(dependent_events);
  if (it->second.deleter) {
    it->second.deleter(data_ptr);
  } else {
    free(data_ptr.ptr);
  }
  lock.lock();
  pool->_allocated -= data_ptr.size;
  pool->_data_ptr_info.erase(it);
  pool->_free_cnt++;
}

void CPUMemoryPool::MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                              const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;

  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& dependent_events = it->second.dependent_events;

  if (stream.device().is_cpu()) {
    dependent_events[stream] = std::make_shared<CPUEvent>(false);
    dependent_events[stream]->Record(stream);
  } else if (stream.device().is_cuda()) {
    // CPU data may be used in host to device copy or device to host copy
    dependent_events[stream] =
      std::make_shared<CUDAEvent>(stream.device(), false);
    dependent_events[stream]->Record(stream);
  } else {
    HT_RUNTIME_ERROR << "CPU arrays must be used on cpu or cuda streams. Got "
                     << stream;
    __builtin_unreachable();
  }
  _mark_cnt++;
}

void CPUMemoryPool::MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                               const Stream& stream) {
  if (stream.is_blocking())
    return;

  std::lock_guard<std::mutex> lock(_mtx);
  // share the event
  std::shared_ptr<Event> event = nullptr;
  if (stream.device().is_cpu()) {
    event = std::make_shared<CPUEvent>(false);
    event->Record(stream);
  } else if (stream.device().is_cuda()) {
    event = std::make_shared<CUDAEvent>(stream.device(), false);
    event->Record(stream);
  } else {
    HT_RUNTIME_ERROR << "CPU arrays must be used on cpu or cuda streams. Got "
                     << stream;
    __builtin_unreachable();
  }
  for (auto& data_ptr : data_ptrs) {
    auto it = _data_ptr_info.find(data_ptr.id);
    HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
      << "Cannot find data " << data_ptr << " from info";
    it->second.dependent_events[stream] = event;
    _mark_cnt++;
  }
}

std::future<void> CPUMemoryPool::WaitDataSpace(DataPtr data_ptr, bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& alloc_stream = it->second.alloc_stream;
  auto& dependent_events = it->second.dependent_events;

  std::future<void> future;
  if (dependent_events.empty()) {
    // Note: Currently the allocation on host memory is blocking, 
    // so we can do nothing here.
    if (async) {
      // avoid future error
      future = std::async([]() {});
    }
  } else if (dependent_events.size() == 1 &&
             dependent_events.begin()->first == alloc_stream) {
    std::shared_ptr<Event>& event = dependent_events.begin()->second;
    if (async) {
      future = std::async([event]() mutable { event->Sync(); });
    } else {
      event->Sync();
    }
  } else {
    if (async) {
      // this will make a copy of `dependent_events`
      future = std::async([dependent_events]() mutable {
        batch_sync_dependent_events(dependent_events);
      });
    } else {
      batch_sync_dependent_events(dependent_events);
    }
  }

  return future;
}

void CPUMemoryPool::PrintSummary() {
  HT_LOG_INFO << name() << ": alloc=" << _allocated << " bytes, "
    << "peak_alloc=" << _peak_allocated << " bytes, "
    << "alloc_cnt=" << _alloc_cnt << ", "
    << "borrow_cnt=" << _borrow_cnt << ", "
    << "free_cnt=" << _free_cnt << ", "
    << "mark_cnt=" << _mark_cnt;
}

namespace {

static std::once_flag cpu_memory_pool_register_flag;

struct CPUMemoryPoolRegister {
  CPUMemoryPoolRegister() {
    std::call_once(cpu_memory_pool_register_flag, []() {
      RegisterMemoryPoolCtor(
          Device(kCPU), []() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<CPUMemoryPool>();
          });
    });
  }
};

static CPUMemoryPoolRegister cpu_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu

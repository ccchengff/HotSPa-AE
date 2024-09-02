#pragma once

#include "hetu/core/device.h"
#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"
#include <string.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>

namespace hetu {
namespace graph {

enum class MEMORY_PROFILE_LEVEL : int8_t {
  MICRO_BATCH = 0,
  INFO,
  WARN
};

class CUDAMemoryInfo {
  public:
    // 单位都是MiB
    size_t mempool_reserved{0};
    size_t mempool_allocated{0};
    size_t all_reserved{0};
    size_t limit{0};
};

class MicroBatchMemoryInfo {
  public:
    // 单位都是MiB
    bool is_forward;
    size_t stage_id;
    size_t micro_batch_id;
    CUDAMemoryInfo begin_memory_info;
    CUDAMemoryInfo end_memory_info;
};

class CUDAProfiler {
  public:
    CUDAProfiler(const Device& device)
    : _device(device) {
      _mempool =  std::dynamic_pointer_cast<hetu::impl::CUDACachingMemoryPool>(GetMemoryPool(device));  
    }
    
    // profile memory
    CUDAMemoryInfo GetCurrMemoryInfo();
    void PrintCurrMemoryInfo(const std::string& prefix);

    // profile NVLink
    void PrintNvlinkStart();
    void PrintNvlinkEnd();

  protected:
    Device _device;

    // profile memory
    std::shared_ptr<hetu::impl::CUDACachingMemoryPool> _mempool;

    // profile NVLink
    unsigned int _device_count = 0; // 有多少个GPU
    std::vector<unsigned int> _nvlink_counts; // 每个GPU有多少条NVLink
    std::vector<std::vector<unsigned long long>> _nvlink_txs; // 记录执行通信代码片段前每个GPU每条NVLink的Raw Tx
    std::vector<std::vector<unsigned long long>> _nvlink_rxs; // 记录执行通信代码片段前每个GPU每条NVLink的Raw Rx
};

std::shared_ptr<CUDAProfiler> GetCUDAProfiler(const Device& device);

std::ostream& operator<<(std::ostream& os, const CUDAMemoryInfo& memory_info);

std::ostream& operator<<(std::ostream& os, const MicroBatchMemoryInfo& micro_batch_memory_info);

class ofstream_sync : public std::ofstream {
  public:
    ofstream_sync(const std::string& filename, std::ios_base::openmode mode = std::ios_base::out)
      : std::ofstream(filename, mode),
        _filename(filename),
        _mode(mode) {
    }

    ~ofstream_sync() {
      this->flush();
      this->close();
      // 再次打开文件并将系统缓存同步到磁盘
      auto fd = ::open(_filename.c_str(), convert_mode(_mode));
      if (fd == -1) {
        HT_RUNTIME_ERROR << "Failed to open file " << _filename;
      }
      if (fsync(fd) == -1) {
        ::close(fd);
        HT_RUNTIME_ERROR << "Failed to fsync file " << _filename;
      }
      ::close(fd);
    }

  private:
    std::string _filename;
    std::ios_base::openmode _mode;

    // 将ios_base::openmode转换为open系统调用使用的标志
    static int convert_mode(std::ios_base::openmode mode) {
      int flags = 0;
      if (mode & std::ios_base::in) flags |= O_RDONLY;
      if (mode & std::ios_base::out) flags |= O_WRONLY | O_CREAT;
      if (mode & std::ios_base::app) flags |= O_APPEND;
      if (mode & std::ios_base::trunc) flags |= O_TRUNC;
      return flags;
    }
};

} // namespace graph
} // namespace hetu
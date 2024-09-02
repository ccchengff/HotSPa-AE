#include "hetu/graph/profiler.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include <nccl.h>
#include <nvml.h>
#include <ctime>
#include <iostream>
#include <fstream>

namespace hetu {
namespace graph {

using hetu::operator<<;

DECLARE_HT_EXCEPTION(nvml_error);

#define NVML_CALL(f)                                                           \
  for (nvmlReturn_t result = (f); result != NVML_SUCCESS; result = NVML_SUCCESS)       \
  __HT_FATAL_SILENT(hetu::graph::nvml_error)                                 \
    << "NVML call " << #f << " failed: " << nvmlErrorString(result)

static std::once_flag nvml_init_flag;
static std::unordered_map<Device, std::shared_ptr<CUDAProfiler>> cuda_memory_profilers;

static void NVML_Init_Once() {
  std::call_once(nvml_init_flag, []() {
    // 初始化NVML库
    NVML_CALL(nvmlInit());
    // register exit handler
    HT_ASSERT(std::atexit([]() {
                NVML_CALL(nvmlShutdown());
              }) == 0)
      << "Failed to register the exit function for NVML.";
  });
}

std::ostream& operator<<(std::ostream& os, const CUDAMemoryInfo& memory_info) {
  os << "{";
  os << "\"mempool reserved\": " << memory_info.mempool_reserved
    << ", \"mempool allocated\": " << memory_info.mempool_allocated
    << ", \"all reserved\": " << memory_info.all_reserved
    << ", \"limit\": " << memory_info.limit;
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const MicroBatchMemoryInfo& micro_batch_memory_info) {
  os << "{" << std::endl;
  os << "\"is forward\": " << micro_batch_memory_info.is_forward << "," << std::endl
    << "\"stage id\": " << micro_batch_memory_info.stage_id << "," << std::endl
    << "\"micro batch id\": " << micro_batch_memory_info.micro_batch_id << "," << std::endl
    << "\"begin memory info\": " << micro_batch_memory_info.begin_memory_info << "," << std::endl
    << "\"end memory info\": " << micro_batch_memory_info.end_memory_info << std::endl,
  os << "}";
  return os;
}

std::shared_ptr<CUDAProfiler> GetCUDAProfiler(const Device& device) {
  auto it = cuda_memory_profilers.find(device);
  if (it == cuda_memory_profilers.end()) {
    auto insertion = cuda_memory_profilers.emplace(device, std::make_shared<CUDAProfiler>(device));
    HT_ASSERT(insertion.second)
      << "Failed to insert the cuda memory profiler for " << device;
    it = insertion.first;
  }
  return it->second;
}

CUDAMemoryInfo CUDAProfiler::GetCurrMemoryInfo() {

  NVML_Init_Once();

  nvmlDevice_t nvml_device;
  nvmlMemory_t memory;
  auto device_index = _device.index();
  NVML_CALL(nvmlDeviceGetHandleByIndex(device_index, &nvml_device));
  NVML_CALL(nvmlDeviceGetMemoryInfo(nvml_device, &memory));

  auto cuda_memory_info = CUDAMemoryInfo();
  cuda_memory_info.limit = memory.total / (1024 * 1024);
  cuda_memory_info.all_reserved = memory.used / (1024 * 1024);
  cuda_memory_info.mempool_allocated = _mempool->GetCurrAllocated() / (1024 * 1024);
  cuda_memory_info.mempool_reserved = _mempool->GetCurrReserved() / (1024 * 1024);
  return cuda_memory_info;
}

void CUDAProfiler::PrintCurrMemoryInfo(const std::string& prefix) {
  
  auto cuda_memory_info = GetCurrMemoryInfo();
  HT_LOG_INFO << "[" << prefix << "] " << _device << ": "
    << "all reserved memory (nvidia-smi) = " << cuda_memory_info.all_reserved << " MiB"
    << ", mempool reserved memory = " << cuda_memory_info.mempool_reserved << " MiB"
    << ", mempool allocated memory = " << cuda_memory_info.mempool_allocated << " MiB";
}

void CUDAProfiler::PrintNvlinkStart() {

  // 只需要一个机器profile即可
  if (hetu::impl::comm::DeviceToWorldRank(_device) != 0) {
    return;
  }
  HT_LOG_INFO << "********* Profile NVLink Start *********";

  // 初始化NVML库
  NVML_Init_Once();
  
  _nvlink_counts.clear();
  _nvlink_txs.clear();
  _nvlink_rxs.clear();

  // 获取GPU数量
  NVML_CALL(nvmlDeviceGetCount(&_device_count));
  for (unsigned int i = 0; i < _device_count; ++i) {
    // Initialization
    _nvlink_counts.emplace_back(0);
    _nvlink_txs.emplace_back();
    _nvlink_rxs.emplace_back();

    // Get current device
    nvmlDevice_t device;
    NVML_CALL(nvmlDeviceGetHandleByIndex(i, &device));
    // Check the NVLink status for each possible link
    for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++) {
      nvmlEnableState_t is_active;
      NVML_CALL(nvmlDeviceGetNvLinkState(device, link, &is_active));
      if (is_active == NVML_FEATURE_ENABLED) {
        _nvlink_counts[i]++;
      }
    }
    HT_LOG_INFO << "GPU " << i << " has " << _nvlink_counts[i] << " NVLink connections active";
    if (_nvlink_counts[i] == 0) {
      continue;
    }

    // 创建NVML字段值数组
    std::vector<nvmlFieldValue_t> field_values(2 * _nvlink_counts[i]);
    for (unsigned int link = 0; link < _nvlink_counts[i]; link++) {
      field_values[2 * link].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
      field_values[2 * link].scopeId = link; // 设置scopeId为linkId
      field_values[2 * link + 1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
      field_values[2 * link + 1].scopeId = link; // 设置scopeId为linkId
    }

    // 记录执行nccl通信代码片段前的NVLink Raw Tx和Raw Rx
    NVML_CALL(nvmlDeviceGetFieldValues(device, field_values.size(), field_values.data()));
    for (unsigned int link = 0; link < _nvlink_counts[i]; link++) {
      _nvlink_txs[i].emplace_back(field_values[2 * link].value.ullVal);
      _nvlink_rxs[i].emplace_back(field_values[2 * link + 1].value.ullVal);
    }
  }
}

void CUDAProfiler::PrintNvlinkEnd() {

  // 只需要一个机器profile即可
  // 如果没有NVLink则不再profile
  if (hetu::impl::comm::DeviceToWorldRank(_device) != 0) {
    return;
  }

  for (unsigned int i = 0; i < _device_count; i++) {
    if (_nvlink_counts[i] == 0) {
      continue;
    }
    // Get current device
    nvmlDevice_t device;
    NVML_CALL(nvmlDeviceGetHandleByIndex(i, &device));

    // 创建NVML字段值数组
    std::vector<nvmlFieldValue_t> field_values(2 * _nvlink_counts[i]);
    for (unsigned int link = 0; link < _nvlink_counts[i]; link++) {
      field_values[2 * link].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
      field_values[2 * link].scopeId = link; // 设置scopeId为linkId
      field_values[2 * link + 1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
      field_values[2 * link + 1].scopeId = link; // 设置scopeId为linkId
    }

    // 获取执行nccl通信代码片段后的NVLink Raw Tx和Raw Rx
    NVML_CALL(nvmlDeviceGetFieldValues(device, field_values.size(), field_values.data()));
    for (unsigned int link = 0; link < _nvlink_counts[i]; link++) {
      _nvlink_txs[i][link] = field_values[2 * link].value.ullVal - _nvlink_txs[i][link];
      _nvlink_rxs[i][link] = field_values[2 * link + 1].value.ullVal - _nvlink_rxs[i][link];
      // 打印NVLink Raw Tx和Raw Rx的变化量
      HT_LOG_INFO << "GPU " << i << " NVLink " << link << " Data Tx Delta: " << _nvlink_txs[i][link] << " KiB";
      HT_LOG_INFO << "GPU " << i << " NVLink " << link << " Data Rx Delta: " << _nvlink_rxs[i][link] << " KiB";
    }
  }
   
  HT_LOG_INFO << "********* Profile NVLink End *********";
}

} // namespace graph
} // namespace hetu

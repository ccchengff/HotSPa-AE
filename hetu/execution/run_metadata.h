#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/autograd/operator.h"
#include "hetu/autograd/autograd.h"
#include "hetu/autograd/topo.h"
#include "hetu/autograd/runtime_context.h"
#include <cuda_runtime.h>

namespace hetu {
namespace execution {

using namespace hetu::autograd;
using namespace std::chrono;

class OpRunMeta;
class RunMetaData;
struct OpInfo;
using OpRunMetaList = std::vector<OpRunMeta>;
using OpRunMetaIdx = std::unordered_map<OpName, int64_t>;
using OpInfoList = std::vector<OpInfo>;

class OpRunMeta {
 public:
  OpRunMeta() = default;

  OpRunMeta(Operator oper) : _op(oper) {}

  void Start() {
    _start = system_clock::now();
    _time_start =
      duration_cast<microseconds>(_start.time_since_epoch()).count();
  }

  void Stop() {
    _stop = system_clock::now();
    _time_stop = duration_cast<microseconds>(_stop.time_since_epoch()).count();
    _duration = _time_stop - _time_start;
  }

  int64_t start() const {
    return _time_start;
  }

  int64_t stop() const {
    return _time_stop;
  }

  int64_t duration() const {
    return _duration;
  }

  OpName name() const {
    return _op->name();
  }

  DeviceIndex device_idx() const {
    return _op->placement().index();
  }

  StreamIndex stream_idx() const {
    return _op->stream().stream_index();
  }

 protected:
  friend std::ostream& operator<<(std::ostream&, const OpRunMeta&);
  Operator _op;
  int64_t _time_start;
  int64_t _time_stop;
  int64_t _duration;
  system_clock::time_point _start;
  system_clock::time_point _stop;
};

std::ostream& operator<<(std::ostream&, const OpRunMeta&);

struct OpInfo {
  std::string ph;
  std::string cat;
  OpName name;
  int64_t pid;
  int64_t tid;
  int64_t ts;
  int64_t dur;
  std::unordered_map<std::string, std::string> args;

  void emplace(std::string key, std::string value) {
    args.emplace(key, value);
  }

  std::string str_args() {
    std::string res = "";
    std::vector<std::string> keys(0), values(0);
    for (auto it = args.begin(); it != args.end(); it++) {
      keys.emplace_back(it->first);
      values.emplace_back(it->second);
    }
    int len = keys.size();
    for (int i = 0; i < len - 1; ++i) {
      res += "\"" + keys[i] + "\":\"" + values[i] + "\",\n";
    }
    res += "\"" + keys[len - 1] + "\":\"" + values[len - 1] + "\"\n";
    return res;
  }
};

class RunMetaData {
 public:
  RunMetaData() {
    op_run_metas = OpRunMetaList(0);
  }

  void Add(OpRunMeta op_run_meta) {
    op_run_metas.emplace_back(op_run_meta);
    op_run_metas_idx.emplace(op_run_meta.name(), op_run_metas.size() - 1);
  }

  OpRunMeta Get(OpName name) {
    return op_run_metas[op_run_metas_idx.find(name)->second];
  }

  int64_t num_ops() const {
    return op_run_metas.size();
  }

  void generate_device_info();

  void generate_operator_info();

  std::string info_to_string(OpInfo info);

  std::string infos_to_string();

  std::string generate_chrome_trace_format();

 protected:
  OpRunMetaList op_run_metas;

  OpRunMetaIdx op_run_metas_idx;

  OpInfoList op_infos;
};

} // namespace execution
} // namespace hetu

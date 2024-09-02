#include "hetu/execution/device_placer.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/execution/run_metadata.h"

namespace hetu {
namespace execution {
void RunMetaData::generate_device_info() {
  for (int i = 0; i < HT_MAX_DEVICE_INDEX; ++i) {
    OpInfo device_info;
    device_info.ph = "M";
    device_info.name = "process_name";
    device_info.pid = i;
    device_info.args = {};
    device_info.emplace("name", "device" + std::to_string(i));
    op_infos.push_back(device_info);
  }
}

void RunMetaData::generate_operator_info() {
  int len = num_ops();
  for (int i = 0; i < len; ++i) {
    OpInfo operator_info;
    operator_info.ph = "X";
    operator_info.cat = "Op";
    operator_info.name = op_run_metas[i].name();
    operator_info.pid = op_run_metas[i].device_idx();
    operator_info.tid = 0;
    operator_info.ts = op_run_metas[i].start();
    operator_info.dur = op_run_metas[i].duration();
    operator_info.args = {};
    operator_info.emplace("name", operator_info.name);
    op_infos.push_back(operator_info);
  }
}

std::string RunMetaData::info_to_string(OpInfo info) {
  std::string res = "";
  if (info.ph == "M") {
    res += "\"name\":\"" + info.name + "\",\n";
    res += "\"ph\":\"" + info.ph + "\",\n";
    res += "\"pid\":" + std::to_string(info.pid) + ",\n";
    res += "\"args\":{\n" + info.str_args() + "}\n";
  } else if (info.ph == "X") {
    res += "\"ph\":\"" + info.ph + "\",\n";
    res += "\"cat\":\"" + info.cat + "\",\n";
    res += "\"name\":\"" + info.name + "\",\n";
    res += "\"pid\":" + std::to_string(info.pid) + ",\n";
    res += "\"tid\":" + std::to_string(info.tid) + ",\n";
    res += "\"ts\":" + std::to_string(info.ts) + ",\n";
    res += "\"dur\":" + std::to_string(info.dur) + ",\n";
    res += "\"args\":{\n" + info.str_args() + "}\n";
  }
  return res;
}

std::string RunMetaData::infos_to_string() {
  int len = op_infos.size();
  std::string res = "";
  for (int i = 0; i < len - 1; ++i) {
    OpInfo info = op_infos[i];
    res += "{\n";
    res += info_to_string(info);
    res += "},\n";
  }
  res += "{\n";
  res += info_to_string(op_infos[len - 1]);
  res += "}\n";
  return res;
}

std::string RunMetaData::generate_chrome_trace_format() {
  op_infos = OpInfoList(0);
  generate_device_info();
  generate_operator_info();
  std::string res = "{\n\"traceEvents\":[\n";
  res += infos_to_string();
  res += "]\n}\n";
  return res;
}

std::ostream& operator<<(std::ostream& os, const OpRunMeta& meta) {
  os << "[" << meta.name() << ", start time:" << meta.start()
     << ", duration:" << meta.duration() << "]";
  return os;
}

} // namespace execution
} // namespace hetu

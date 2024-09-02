#include "hetu/graph/autocast/autocast.h"
#include <thread>

namespace hetu {
namespace graph {

std::once_flag AutoCast::_init_flag;
std::vector<std::shared_ptr<AutoCast>> AutoCast::_autocasts;
std::shared_ptr<AutoCast> AutoCast::_default_autocast;
thread_local std::stack<AutoCastId> AutoCast::_cur_autocast_ctx;

void AutoCast::Init() {
  auto concurrency = std::thread::hardware_concurrency();
  AutoCast::_autocasts.reserve(MIN(concurrency, 16) * 2);
  AutoCast::_default_autocast = AutoCast::MakeAutoCast(true);
}

AutoCastId AutoCast::_next_autocast_id() {
  static std::atomic<AutoCastId> _global_autocast_id{0};
  return _global_autocast_id++;
}

DataType AutoCast::WidestType(const TensorList& inputs) {
  DataType widest_type = DataType::FLOAT16;
  for (const auto& input: inputs) {
    if (input->dtype() == DataType::FLOAT64)
      return DataType::FLOAT64;
    if (input->dtype() == DataType::FLOAT32)
      widest_type = DataType::FLOAT32;
  }
  return widest_type;
}

void AutoCast::Tensor_AutoCast(TensorList& inputs, DataType datatype) {
}

void AutoCast::Graph_AutoCast(TensorList& inputs, Operator op) {
  auto optype = op->type();
  if (is_optimizer_update_op(op) || optype == "DataTransfer" ||
      optype == "DataTransferOp" || optype == "DataH2DOp" ||
      optype == "DataD2HOp") 
    return;
  auto autocast_id = AutoCast::cur_autocast_ctx();
  if (autocast_id == UINT64_MAX)
    return;
  auto autocast = AutoCast::GetAutoCast(autocast_id);
  if (!autocast.enabled())
    return;
  DataType datatype = DataType::UNDETERMINED;
  if (autocast.cast_type() != DataType::UNDETERMINED)
    datatype = autocast.cast_type();
  else {
  }
  if (datatype != DataType::UNDETERMINED) {
    for (auto& input: inputs) {
      if (input->dtype() != datatype && 
          (input->dtype() == DataType::BFLOAT16 ||
          input->dtype() == DataType::FLOAT16 ||
          input->dtype() == DataType::FLOAT32 ||
          input->dtype() == DataType::FLOAT64)) {
        auto& input_graph = Graph::GetGraph(input->graph_id());
        HT_LOG_DEBUG << input->producer()->type() << " " << input->dtype();
        input = MakeDataTransferOp(datatype, input);
      }
    }
  }
}

void AutoCast::Graph_AutoCast(TensorList& inputs, std::shared_ptr<OpInterface> body) {
  auto optype = body->type();
  if (is_optimizer_update_op(*body) || optype == "DataTransfer" || optype == "DataTransferOp" || optype == "DataH2DOp" || optype == "DataD2HOp") 
    return;
  auto autocast_id = AutoCast::cur_autocast_ctx();
  if (autocast_id == UINT64_MAX)
    return;
  auto autocast = AutoCast::GetAutoCast(autocast_id);
  if (!autocast.enabled())
    return;
  DataType datatype = DataType::UNDETERMINED;
  if (autocast.cast_type() != DataType::UNDETERMINED)
    datatype = autocast.cast_type();
  if (datatype != DataType::UNDETERMINED) {
    for (auto& input: inputs) {
      if (input->dtype() != datatype && 
          (input->dtype() == DataType::BFLOAT16 ||
          input->dtype() == DataType::FLOAT16 ||
          input->dtype() == DataType::FLOAT32 ||
          input->dtype() == DataType::FLOAT64)) {
        auto& input_graph = Graph::GetGraph(input->graph_id());
        if (input_graph.type() == GraphType::EXECUTABLE ||
            input_graph.type() == GraphType::EAGER) {
          input = MakeDataTransferOp(datatype, input);
        }
      }
    }
  }
}

} // namespace graph
} // namespace hetu
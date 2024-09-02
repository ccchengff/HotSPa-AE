#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/autograd/op_meta.h"
#include "hetu/autograd/tensor.h"
#include "hetu/common/macros.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include "hetu/autograd/runtime_context.h"
#include <functional>

namespace hetu {
namespace autograd {

class OperatorPool final {
 private:
  template <typename OpDef>
  friend class OpWrapper;
  friend class OperatorDef;
  friend class TensorDef;
  friend class Tensor;
  static std::unordered_map<OpId, Operator> _global_ops_from_ids;
  static std::unordered_map<OpName, Operator> _global_ops_from_names;
  static std::unordered_map<OpName, uint32_t> _global_op_name_counts;
  static void RegisterOp(Operator& op);
  static OpName CreateAvailableOpName(const OpType& op_type, OpId id,
                                      const OpName& optional_name);

 public:
  static Operator& GetOp(OpId id);
  static Operator& GetOp(const OpName& name);
  static OpList GetAllOps();
};

/******************************************************
 * Op Definition
 ******************************************************/
class OperatorDef : public shared_ptr_target {
 protected:
  friend class TensorDef;
  friend class Tensor;
  template <typename T>
  friend class OpWrapper;

  OperatorDef(OpType&& op_type, const TensorList& inputs,
              const OpMeta& op_meta);

 public:
  ~OperatorDef() = default;

  // disable copy constructor and move constructor
  OperatorDef(const OperatorDef&) = delete;
  OperatorDef& operator=(const OperatorDef&) = delete;
  OperatorDef(OperatorDef&&) = delete;
  OperatorDef& operator=(OperatorDef&&) = delete;

  NDArrayList Compute(const NDArrayList& inputs, RuntimeContext& ctx);

  TensorList Gradient(const TensorList& grad_outputs);

  HTShapeList InferShape(const HTShapeList& input_shapes);

  bool MapToParallelDevices(const DeviceGroup& placement_group);

  bool PlaceToLocalDevice(const Device& placement, StreamIndex stream_id);

  inline void Sync() {
    _stop->Sync();
  }

  void MarkAsComputed(const NDArrayList& output_vals);

  inline int64_t TimeCost() {
    return _stop->TimeSince(*_start);
  }

  OpId id() const noexcept {
    return _id;
  }

  const OpType& type() const noexcept {
    return _type;
  }

  const OpName& name() const noexcept {
    return _op_meta.name;
  }

  OpName grad_name(size_t input_id = 0) const {
    return name() + "_grad_" + input(input_id)->name();
  }

  const DeviceGroup& device_group() const noexcept {
    return _op_meta.device_group;
  }

  StreamIndex stream_index() const noexcept {
    return _stream.is_defined() ? _stream.stream_index()
                                : _op_meta.stream_index;
  }

  const OpMeta& op_meta() const noexcept {
    return _op_meta;
  }

  OpMeta grad_op_meta() const {
    return OpMeta()
      .set_stream_index(stream_index())
      .set_device_group(device_group());
  }

  virtual bool inplace() const {
    return false;
  }

  const TensorList& inputs() const noexcept {
    return _inputs;
  }

  const TensorList& outputs() const noexcept {
    return _outputs;
  }

  const Tensor& input(size_t i) const {
    return _inputs[i];
  }

  Tensor& input(size_t i) {
    return _inputs[i];
  }

  const Tensor& output(size_t i) const {
    return _outputs[i];
  }

  Tensor& output(size_t i) {
    return _outputs[i];
  }

  size_t num_inputs() const {
    return _inputs.size();
  }

  size_t num_outputs() const {
    return _outputs.size();
  }

  const TensorList& in_dep_linkers() const noexcept {
    return _extra_in_dep_linkers;
  }

  const Tensor& out_dep_linker() const noexcept {
    return _extra_out_dep_linker;
  }

  Tensor& out_dep_linker() noexcept {
    return _extra_out_dep_linker;
  }

  size_t in_degrees() const {
    return num_inputs() + _extra_in_dep_linkers.size();
  }

  size_t out_degrees() const {
    size_t ret = _extra_out_dep_linker->num_consumers();
    for (auto& output : _outputs)
      ret += output->num_consumers();
    return ret;
  }

  OpRefList input_ops_ref() {
    OpRefList ret;
    ret.reserve(in_degrees());
    for (auto& e : _inputs)
      ret.push_back(e->producer());
    for (auto& e : _extra_in_dep_linkers)
      ret.push_back(e->producer());
    return ret;
  }

  OpRefList output_ops_ref() {
    OpRefList ret;
    ret.reserve(out_degrees());
    for (auto& e : _outputs)
      ret.insert(ret.end(), e->_consumers.begin(), e->_consumers.end());
    ret.insert(ret.end(), _extra_out_dep_linker->_consumers.begin(),
               _extra_out_dep_linker->_consumers.end());
    return ret;
  }

  virtual uint64_t op_indicator() const noexcept {
    return 0;
  }

  const DeviceGroup& placement_group() const noexcept {
    return _placement_group;
  }

  const Device& placement() const noexcept {
    return _placement;
  }

  const Stream& stream() const noexcept {
    return _stream;
  }

  bool is_computed() const {
    return _computed;
  }
  
  void ReplaceInput(size_t index, Tensor new_input); // 暂时先挪到public来
  void AddInDeps(const TensorList& in_deps);

 protected:
  // Walkaround methods to get the corresponding wrapper
  inline Operator& get_self() {
    return out_dep_linker()->producer();
  }

  inline const Operator& get_self() const {
    return out_dep_linker()->producer();
  }

  inline void AddOutput(const NDArrayMeta& output_meta) {
    HT_RUNTIME_ERROR_IF(output_meta.dtype == kUndeterminedDataType)
      << "Data type is not provided for output " << _outputs.size()
      << " of the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\").";
    _outputs.emplace_back(name() + ":" + std::to_string(_outputs.size()),
                          static_cast<int32_t>(_outputs.size()), output_meta);
  }

  inline void AddOutputs(const std::vector<NDArrayMeta>& output_meta_list) {
    for (const auto& output_meta : output_meta_list)
      AddOutput(output_meta);
  }

  virtual NDArrayList DoCompute(const NDArrayList& inputs,
                                RuntimeContext& ctx) {
    NDArrayList outputs = DoAllocOutputs(inputs, ctx);
    DoCompute(inputs, outputs, ctx);
    return outputs;
  }

  virtual void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                         RuntimeContext& ctx) {
    HT_NOT_IMPLEMENTED << "Compute fn of op \"" << type()
                       << "\" is not defined";
  }

  virtual TensorList DoGradient(const TensorList& grad_outputs) {
    HT_NOT_IMPLEMENTED << "Gradient fn of op \"" << type()
                       << "\" is not defined";
    return {};
  }

  virtual void DoInferMeta() {
    HT_NOT_IMPLEMENTED << "InferMeta fn of op \"" << type()
                       << "\" is not defined";
  }

  virtual HTShapeList DoInferShape(const HTShapeList& input_shapes) {
    if (num_outputs() == 0)
      return {};
    HT_NOT_IMPLEMENTED << "InferShape fn of op \"" << type()
                       << "\" is not defined";
    return {};
  }

  virtual void DoDeduceStates();

  virtual NDArrayList DoAllocOutputs(const NDArrayList& inputs,
                                     RuntimeContext& ctx);

  virtual bool DoMapToParallelDevices(const DeviceGroup& placement_group);

  virtual bool DoPlaceToLocalDevice(const Device& placement,
                                    StreamIndex stream_id);

  inline void CheckNumInputsEqual(size_t num) const {
    HT_ASSERT_EQ(num, num_inputs())
      << "Op \"" << type() << "\" "
      << "expected to take " << num_inputs() << " inputs, "
      << "got " << num;
  }

  inline void CheckNumOutputsEqual(size_t num) const {
    HT_ASSERT_EQ(num, num_outputs())
      << "Op \"" << type() << "\" "
      << "expected to take " << num_outputs() << " outputs, "
      << "got " << num;
  }

  void BlockOrSync(Tensor& dep);

  // metadata
  const OpId _id;
  const OpType _type;
  OpMeta _op_meta;

  // related edges
  TensorList _inputs;
  TensorList _outputs;
  TensorList _extra_in_dep_linkers;
  Tensor _extra_out_dep_linker;

  // runtime states
  DeviceGroup _placement_group;
  Device _placement;
  Stream _stream;
  std::shared_ptr<Event> _start;
  std::shared_ptr<Event> _stop;

  // for define-by-run mode
  bool _computed{false};

 private:
  static OpId _next_op_id() {
    static std::atomic<OpId> _global_op_id(0);
    return _global_op_id++;
  }
};

template <typename OpDef>
class OpWrapper : public shared_ptr_wrapper<OpDef> {
 protected:
  template <typename DerivedOpDef>
  OpWrapper(std::shared_ptr<DerivedOpDef> ptr)
  : shared_ptr_wrapper<DerivedOpDef>() {
    static_assert(std::is_base_of<OpDef, DerivedOpDef>::value,
                  "Tempalte DerivedOpDef is not derived from template OpDef.");
    this->_ptr = ptr;
    HT_ASSERT(this->_ptr) << "Passing a nullptr of OperatorDef "
                          << "to the constructor of Operator is not allowed. "
                          << "If you wish to declare an empty operator, "
                          << "call Operator() instead.";
    OperatorPool::RegisterOp(reinterpret_cast<Operator&>(*this));

    // Note: Since we cannot retrive `Operator` inside the constructor of
    // `OperatorDef`, we defer the assignment to producer and consumers here.
    auto& self = reinterpret_cast<Operator&>(*this);
    for (auto& input : this->_ptr->_inputs)
      input->AddConsumer(self);
    for (auto& in_dep : this->_ptr->_extra_in_dep_linkers)
      in_dep->AddConsumer(self);
    for (auto& output : this->_ptr->_outputs)
      output->SetProducer(self);
    this->_ptr->_extra_out_dep_linker->SetProducer(self);
  }

 public:
  OpWrapper() = default;
  using shared_ptr_wrapper<OpDef>::operator=;
  template <typename DerivedOpDef>
  friend class OpWrapper;
  template <typename DerivedOpDef>
  OpWrapper(const OpWrapper<DerivedOpDef>& op) : shared_ptr_wrapper<OpDef>() {
    static_assert(std::is_base_of<OpDef, DerivedOpDef>::value,
                  "Tempalte DerivedOpDef is not derived from template OpDef");
    this->_ptr = op._ptr;
  }
};

/******************************************************
 * Indicators of Operators
 ******************************************************/

static const uint64_t DATA_LOADER_OP = 1ul;
static const uint64_t PLACEHOLDER_OP = 1ul << 1;
static const uint64_t VARIABLE_OP = 1ul << 2;
static const uint64_t HOST_TO_DEVICE_OP = 1ul << 3;
static const uint64_t DEVICE_TO_HOST_OP = 1ul << 4;
static const uint64_t PEER_TO_PEER_SEND_OP = 1ul << 5;
static const uint64_t PEER_TO_PEER_RECV_OP = 1ul << 6;
static const uint64_t ALL_TO_ALL_OP = 1ul << 7;
static const uint64_t ALL_REDUCE_OP = 1ul << 8;
static const uint64_t ALL_GATHER_OP = 1ul << 9;
static const uint64_t REDUCE_SCATTER_OP = 1ul << 10;
static const uint64_t BROADCAST_OP = 1ul << 11;
static const uint64_t REDUCE_OP = 1ul << 12;
static const uint64_t P2P_OP = 1ul << 13;
static const uint64_t BATCHED_ISEND_IRECV_OP = 1ul << 14;
static const uint64_t GATHER_OP = 1ul << 15;
static const uint64_t SCATTER_OP = 1ul << 16;
static const uint64_t COMM_SPLIT_OP = 1ul << 19;
static const uint64_t COMM_OP = 1ul << 20;
static const uint64_t UNKNOWN_OP = 1ul << 21;
static const uint64_t SPLIT_OP = 1ul << 61;
static const uint64_t OPTIMIZER_UPDATE_OP = 1ul << 62;
static const uint64_t GROUP_OP = 1ul << 63;

inline bool is_comm_op(const Operator& op) {
  return (op->op_indicator() & COMM_OP) != 0;
}
inline bool is_data_loader_op(const Operator& op) {
  return (op->op_indicator() & DATA_LOADER_OP) != 0;
}
inline bool is_placeholder_op(const Operator& op) {
  return (op->op_indicator() & PLACEHOLDER_OP) != 0;
}
inline bool is_variable_op(const Operator& op) {
  return (op->op_indicator() & VARIABLE_OP) != 0;
}
bool is_trainable_op(const Operator& op);
inline bool is_host_to_device_op(const Operator& op) {
  return (op->op_indicator() & HOST_TO_DEVICE_OP) != 0;
}
inline bool is_device_to_host_op(const Operator& op) {
  return (op->op_indicator() & DEVICE_TO_HOST_OP) != 0;
}
inline bool is_peer_to_peer_send_op(const Operator& op) {
  return (op->op_indicator() & PEER_TO_PEER_SEND_OP) != 0;
}
inline bool is_peer_to_peer_recv_op(const Operator& op) {
  return (op->op_indicator() & PEER_TO_PEER_RECV_OP) != 0;
}
inline bool is_all_to_all_op(const Operator& op) {
  return (op->op_indicator() & ALL_TO_ALL_OP) != 0;
}
inline bool is_all_reduce_op(const Operator& op) {
  return (op->op_indicator() & ALL_REDUCE_OP) != 0;
}
inline bool is_batched_isend_irecv_op(const Operator& op) {
  return (op->op_indicator() & BATCHED_ISEND_IRECV_OP) != 0;
}
inline bool is_communucation_op(const Operator& op) {
  return is_peer_to_peer_send_op(op) || is_peer_to_peer_recv_op(op) ||
    is_all_to_all_op(op) || is_all_reduce_op(op);
}
inline bool is_split_op(const Operator& op) {
  return (op->op_indicator() & SPLIT_OP) != 0;
}
inline bool is_optimizer_update_op(const Operator& op) {
  return (op->op_indicator() & OPTIMIZER_UPDATE_OP) != 0;
}
inline bool is_group_op(const Operator& op) {
  return (op->op_indicator() & GROUP_OP) != 0;
}

/******************************************************
 * Logging & Streaming
 ******************************************************/

std::ostream& operator<<(std::ostream&, const Operator&);

} // namespace autograd
} // namespace hetu

namespace std {
inline std::string to_string(const hetu::autograd::Operator& op) {
  std::ostringstream os;
  os << op;
  return os.str();
}
} // namespace std

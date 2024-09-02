#include "hetu/autograd/operator.h"
#include "hetu/autograd/ops/DataTransfer.h"
#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/ops/Group.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"

namespace hetu {
namespace autograd {

std::unordered_map<OpId, Operator> OperatorPool::_global_ops_from_ids;
std::unordered_map<OpName, Operator> OperatorPool::_global_ops_from_names;
std::unordered_map<OpName, uint32_t> OperatorPool::_global_op_name_counts;

void OperatorPool::RegisterOp(Operator& op) {
  HT_ASSERT(op.is_defined())
    << "It is not allowed to register an empty operation.";
  OpId id = op->id();
  const OpName& name = op->name();
  HT_LOG_TRACE << "Registering op with id " << id << " and name \"" << name
               << "\"...";
  HT_ASSERT(OperatorPool::_global_ops_from_ids.find(id) ==
            OperatorPool::_global_ops_from_ids.end())
    << "Operator with id \"" << id << "\" "
    << "has already been registered.";
  HT_ASSERT(OperatorPool::_global_ops_from_names.find(name) ==
            OperatorPool::_global_ops_from_names.end())
    << "Operator with name \"" << name << "\" "
    << "has already been registered.";
  OperatorPool::_global_ops_from_ids[id] = op;
  OperatorPool::_global_ops_from_names[name] = op;
}

OpName OperatorPool::CreateAvailableOpName(const OpType& op_type, OpId id,
                                           const OpName& optional_name) {
  if (optional_name.empty()) {
    return op_type + '(' + std::to_string(id) + ')';
  } else {
    auto it = OperatorPool::_global_op_name_counts.find(optional_name);
    if (it == OperatorPool::_global_op_name_counts.end()) {
      OperatorPool::_global_op_name_counts[optional_name] = 1;
      return optional_name;
    } else {
      return optional_name + '_' + std::to_string(it->second++);
    }
  }
}

Operator& OperatorPool::GetOp(OpId id) {
  return OperatorPool::_global_ops_from_ids[id];
}

Operator& OperatorPool::GetOp(const OpName& name) {
  return OperatorPool::_global_ops_from_names[name];
}

OpList OperatorPool::GetAllOps() {
  // TODO: garbage collection
  OpList ret;
  for (auto& kv : _global_ops_from_ids)
    ret.push_back(kv.second);
  return ret;
}

OperatorDef::OperatorDef(OpType&& op_type, const TensorList& inputs,
                         const OpMeta& op_meta)
: _id{_next_op_id()},
  _type{std::move(op_type)},
  _op_meta(op_meta),
  _inputs(inputs) {
  _op_meta.set_name(
    OperatorPool::CreateAvailableOpName(_type, _id, _op_meta.name));
  // All inputs must be tensors
  for (size_t i = 0; i < _inputs.size(); i++) {
    HT_ASSERT(_inputs[i].is_defined() && _inputs[i]->is_tensor())
      << "Failed to construct the \"" << _type << "\" operation "
      << "(with name \"" << _op_meta.name << "\"): "
      << "Cannot convert input " << i << " to a tensor: " << _inputs[i] << ".";
  }
  // Extra input depencenies. May be tensors or output dependency linkers
  auto& extra_deps = _op_meta.extra_deps;
  if (extra_deps.empty() || _type == quote(GroupOp)) {
    // Walkaround: if we are constructing a group op,
    // we shall not construct another group op to handle the dependecies.
    _extra_in_dep_linkers.reserve(extra_deps.size());
    for (auto& dep : extra_deps) {
      _extra_in_dep_linkers.push_back(dep);
    }
  } else if (extra_deps.size() == 1) {
    _extra_in_dep_linkers.push_back(extra_deps.front());
  } else {
    // Merge dependencies into a group op
    auto in_dep =
      GroupOp(extra_deps, OpMeta().set_name(_op_meta.name + "_in_dep"))
        ->out_dep_linker();
    _extra_in_dep_linkers.push_back(in_dep);
  }
  // The output dependency linker of this op
  _extra_out_dep_linker = Tensor(_op_meta.name + "_out_dep", -1);
}

NDArrayList OperatorDef::Compute(const NDArrayList& inputs,
                                 RuntimeContext& ctx) {
  HT_LOG_TRACE << "Calling Compute fn of op \"" << name() << "\"...";
  CheckNumInputsEqual(inputs.size());
  for (auto& input : _inputs)
    BlockOrSync(input);
  for (auto& in_dep : _extra_in_dep_linkers)
    BlockOrSync(in_dep);
  _start->Record(_stream);
  NDArrayList ret = DoCompute(inputs, ctx);
  _stop->Record(_stream);
  return ret;
}

TensorList OperatorDef::Gradient(const TensorList& grad_outputs) {
  HT_LOG_TRACE << "Calling Gradient fn of op \"" << name() << "\"...";
  CheckNumOutputsEqual(grad_outputs.size());
  return DoGradient(grad_outputs);
}

HTShapeList OperatorDef::InferShape(const HTShapeList& input_shapes) {
  HT_LOG_TRACE << "Calling InferShape fn of op \"" << name() << "\"...";
  CheckNumInputsEqual(input_shapes.size());
  return DoInferShape(input_shapes);
}

NDArrayList OperatorDef::DoAllocOutputs(const NDArrayList& inputs,
                                        RuntimeContext& ctx) {
  NDArrayList outputs;
  outputs.reserve(num_outputs());
  if (num_outputs() > 0) {
    HTShapeList input_shapes;
    input_shapes.reserve(inputs.size());
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(input_shapes),
                   [](const NDArray& input) { return input->shape(); });
    // Note: InferShape is still necessary in pipeline parallelism
    HTShapeList output_shapes = InferShape(input_shapes);
    for (size_t i = 0; i < num_outputs(); i++) {
      if (_outputs[i]->_data.is_defined()) {
        HT_RUNTIME_ERROR_IF(_outputs[i]->_data->shape() != output_shapes[i])
          << "Tensor bound to data with shape " << _outputs[i]->_data->shape()
          << " (expected " << output_shapes[i] << ")";
        outputs.push_back(_outputs[i]->_data);
      } else {
        outputs.push_back(
          NDArray::empty(output_shapes[i], placement(), _outputs[i]->dtype()));
      }
    }
  }
  return outputs;
}

bool OperatorDef::MapToParallelDevices(const DeviceGroup& placement_group) {
  HT_LOG_TRACE << "Mapping op \"" << name() << "\" to " << placement_group
               << "...";
  // TODO: check whether the placement group is valid
  for (auto& input : _inputs) {
    auto& input_op = input->producer();
    HT_ASSERT(!input_op->placement_group().empty())
      << "Device mapping must be done in topo order. "
      << "Operator \"" << input_op->name() << "\" "
      << "is not properly mapped before "
      << "its follow-up operator \"" << name() << "\".";
  }
  return DoMapToParallelDevices(placement_group);
}

bool OperatorDef::DoMapToParallelDevices(const DeviceGroup& placement_group) {
  _placement_group = placement_group;
  // TODO: set the parallel statuses of outputs
  for (auto& output : _outputs) {
    output->set_placement_group(placement_group);
  }

  // add P2P communication ops
  auto& dst_group = _placement_group;
  for (size_t i = 0; i < _inputs.size(); i++) {
    auto& input_op = _inputs[i]->producer();
    const auto& src_group = input_op->placement_group();
    if (src_group != dst_group) {
      HT_ASSERT(src_group.num_devices() == dst_group.num_devices())
        << "Currently we require equal data parallelism degree "
        << "between pipeline stages (got " << src_group.num_devices() << " vs. "
        << dst_group.num_devices();

      // remove redundant P2P ops if possible
      auto& input_op = _inputs[i]->producer();
      if (is_peer_to_peer_recv_op(input_op)) {
        const auto& p2p_recv_op = reinterpret_cast<const P2PRecvOp&>(input_op);
        if (p2p_recv_op->src_group() == dst_group) {
          ReplaceInput(i, p2p_recv_op->send_op()->output(0));
          continue;
        }
      }

      // re-use P2P ops if possible
      bool reused = false;
      for (size_t j = 0; j < _inputs[i]->num_consumers(); j++) {
        auto& consumer_op = _inputs[i]->consumer(j);
        if (consumer_op->id() != _id && is_peer_to_peer_send_op(consumer_op)) {
          const auto& p2p_send_op =
            reinterpret_cast<const P2PSendOp&>(consumer_op);
          if (p2p_send_op->dst_group() == dst_group) {
            ReplaceInput(i, p2p_send_op->recv_op()->output(0));
            reused = true;
            break;
          }
        }
      }
      if (reused)
        continue;

      // create P2P ops if needed
      P2PSendOp p2p_send_op(_inputs[i], dst_group);
      P2PRecvOp p2p_recv_op(src_group, _inputs[i]->dtype(),
                            _inputs[i]->shape());
      p2p_send_op->MapToParallelDevices(src_group);
      p2p_recv_op->MapToParallelDevices(dst_group);
      p2p_send_op->BindRecvOp(p2p_recv_op);
      p2p_recv_op->BindSendOp(p2p_send_op);
      p2p_recv_op->output(0)->set_placement_group(placement_group);
      ReplaceInput(i, p2p_recv_op->output(0));
    }
  }
  return true;
}

bool OperatorDef::PlaceToLocalDevice(const Device& placement,
                                     StreamIndex stream_id) {
  HT_LOG_TRACE << "Placing op \"" << name() << "\" to " << placement << "...";
  HT_ASSERT(!placement.is_undetermined()) << "Placement must be determined";
  for (auto& input : _inputs) {
    HT_ASSERT(!input->placement().is_undetermined())
      << "Device placement must be done in topo order. "
      << "Operator \"" << input->producer()->name() << "\" "
      << "is not properly placed before "
      << "its follow-up operator \"" << name() << "\".";
  }
  if (!_placement_group.empty() && !_placement_group.contains(placement))
    return false;
  return DoPlaceToLocalDevice(placement, stream_id);
}

bool OperatorDef::DoPlaceToLocalDevice(const Device& placement,
                                       StreamIndex stream_id) {
  _placement = placement;

  // set up stream and events
  _stream = Stream(_placement, stream_id);
  if (_placement.is_cuda()) {
    _start = std::make_shared<hetu::impl::CUDAEvent>(_placement);
    _stop = std::make_shared<hetu::impl::CUDAEvent>(_placement);
  } else {
    _start = std::make_shared<hetu::impl::CPUEvent>(_placement);
    _stop = std::make_shared<hetu::impl::CPUEvent>(_placement);
  }
  
  // 顺便给tensor的distributed_states也增加placement的attribute
  for (auto& output : _outputs)
    output->set_placement(placement);

  // add transfer ops
  for (size_t i = 0; i < _inputs.size(); i++) {
    if (_inputs[i]->placement() != _placement) {
      HT_RUNTIME_ERROR_IF(!_inputs[i]->placement().local())
        << "Please use P2P communication to fetch remote input";

      // remove redundant H2D or D2H ops if possible
      auto& input_op = _inputs[i]->producer();
      if ((is_host_to_device_op(input_op) || is_device_to_host_op(input_op)) &&
          input_op->input(0)->placement() == _placement) {
        ReplaceInput(i, input_op->input(0));
        continue;
      }

      // re-use H2D or D2H ops if possible
      bool reused = false;
      for (size_t j = 0; j < _inputs[i]->num_consumers(); j++) {
        auto& consumer_op = _inputs[i]->consumer(j);
        if (consumer_op->id() != _id &&
            (is_host_to_device_op(consumer_op) ||
             is_device_to_host_op(consumer_op)) &&
            consumer_op->placement() == _placement) {
          ReplaceInput(i, consumer_op->output(0));
          reused = true;
          break;
        }
      }
      if (reused)
        continue;

      // create H2D or D2H ops if needed
      Operator transfer_op;
      StreamIndex transfer_stream_id = kBlockingStream;
      if (_inputs[i]->placement().is_cpu()) {
        transfer_op = DataH2DOp(_inputs[i], _placement);
        transfer_stream_id = kH2DStream;
      } else if (_placement.is_cpu()) {
        transfer_op = DataD2HOp(_inputs[i]);
        transfer_stream_id = kD2HStream;
      } else {
        // TODO: support cuda memcpy across processes
        HT_NOT_IMPLEMENTED << "We should use NCCL for P2P communication.";
      }
      if (!input_op->placement_group().empty())
        transfer_op->MapToParallelDevices(input_op->placement_group());
      transfer_op->PlaceToLocalDevice(_placement, transfer_stream_id);
      ReplaceInput(i, transfer_op->output(0));
    }
  }

  return true;
}

void OperatorDef::DoDeduceStates() {
  // default: distributed states of output tensor directly copy from input tensor
  // check input states is valid & check distributed states of all input tensor are the same.
  HT_LOG_DEBUG << name() << ": default copy states from inputs";
  DistributedStates default_ds;
  for (auto& input : _inputs) {
    auto input_ds = input->get_distributed_states(); 
    HT_ASSERT(input_ds.is_valid()) << name() << ": input states must be valid!";
    HT_ASSERT(input_ds.get_dim(-2) == 1) << name() << ": input shouldn't be partial!";      
    if (!default_ds.is_valid()) {
      default_ds.set_distributed_states(input_ds);
    } else {
      HT_ASSERT(default_ds.check_equal(input_ds))
        << name() << ": in default DoDeduceStates: distributed states of all input tensor must be same!"
        << ", " << default_ds.ds_info() << " vs " << input_ds.ds_info();
    }
  }
  for (auto& output : _outputs) {
    output->set_distributed_states(default_ds);
  }
}

void OperatorDef::BlockOrSync(Tensor& dep) {
  if (!dep.is_defined())
    return;
  if (is_placeholder_op(dep->producer()))
    return;
  auto& dep_op = dep->producer();
  if (dep_op->placement() != _placement) {
    // Question: In pipeline parallel, should we take extra dependencies
    // across pipeline stages into account?
    return;
  }
  if (dep_op->_stream != _stream) {
    // Both ops are on CUDA or on CPU. We can block the current op
    // by waiting for the stop event of the dependency.
    if (dep_op->_stream.device_type() == _stream.device_type()) {
      dep_op->_stop->Block(_stream);
    } else {
      // We cannot block on different device types. Just sync here.
      dep_op->_stop->Sync();
    }
  }
}

void OperatorDef::ReplaceInput(size_t index, Tensor new_input) {
  auto& old_input = _inputs[index];
  old_input->_consumers.erase(
    std::remove_if(old_input->_consumers.begin(), old_input->_consumers.end(),
                   [&](const Operator& op) { return op->id() == _id; }));
  _inputs[index] = new_input;
  new_input->AddConsumer(get_self());
}

void OperatorDef::AddInDeps(const TensorList& in_deps) {
  if (in_deps.empty()) {
    return;
  }
  if (in_deps.size() == 1) {
    _extra_in_dep_linkers.push_back(in_deps.front());
  } else {
    auto in_dep = GroupOp(in_deps, OpMeta().set_name(_op_meta.name + "_extra_in_dep"))->out_dep_linker();
    _extra_in_dep_linkers.push_back(in_dep);
  }
  _extra_in_dep_linkers.back()->AddConsumer(get_self());
}

void OperatorDef::MarkAsComputed(const NDArrayList& output_vals) {
  CheckNumOutputsEqual(output_vals.size());
  _computed = true;
  for (size_t i = 0; i < num_outputs(); i++) {
    _outputs[i]->_computed = true;
    _outputs[i]->_data = output_vals[i];
  }
  _extra_out_dep_linker->_computed = true;
}

bool is_trainable_op(const Operator& op) {
  return is_variable_op(op) &&
    reinterpret_cast<const VariableOp&>(op)->trainable();
}

std::ostream& operator<<(std::ostream& os, const Operator& op) {
  if (op.is_defined())
    os << op->name();
  else
    os << "Operator()";
  return os;
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

uint64_t CommOpDef::get_comm_type() {
  // TODO: 后面看要不要把单纯的get拆出一个函数来
  auto src_ds = _inputs[0]->get_distributed_states();
  auto dst_ds = _dst_distributed_states;
  if (src_ds.check_pure_duplicate()) {
    _comm_type = COMM_SPLIT_OP;
    HT_LOG_DEBUG << "COMM_SPLIT_OP";
  } else if (src_ds.check_allreduce(dst_ds)) {
    _comm_type = ALL_REDUCE_OP;
    HT_LOG_DEBUG << "ALL_REDUCE_OP";
  } else if (src_ds.check_allgather(dst_ds)) {
    _comm_type = ALL_GATHER_OP;
    HT_LOG_DEBUG << "ALL_GATHER_OP";
  } else if (src_ds.check_reducescatter(dst_ds)) {
    _comm_type = REDUCE_SCATTER_OP;
    HT_LOG_DEBUG << "REDUCE_SCATTER_OP";
  } else {
    _comm_type = P2P_OP; // other case: 非集合通信, 部分device之间p2p通信
    HT_LOG_DEBUG << "P2P_OP";
  }
  return _comm_type;
}  

// devices by dim for collective communication
DeviceGroup CommOpDef::get_devices_by_dim(int32_t dim) {
  HT_ASSERT(!_placement_group.empty()) << "Placement Group should be assigned before get devices by dim " << dim;
  int32_t local_device_idx = _placement_group.get_index(_placement);
  auto src_ds = _inputs[0]->get_distributed_states();
  auto order = src_ds.get_order();
  auto states = src_ds.get_states();

  auto idx = std::find(order.begin(), order.end(), dim);
  int32_t interval = 1;
  for (auto cur_order = idx + 1; cur_order != order.end(); cur_order++) {
    interval *= states[*cur_order];
  }
  int32_t macro_interval = interval * src_ds.get_dim(dim);
  int32_t start = local_device_idx - local_device_idx % macro_interval + local_device_idx % interval;
  std::vector<Device> comm_group;
  for (auto i = start; i < start + macro_interval; i += interval) {
    comm_group.push_back(_placement_group.get(i));
  }
  return DeviceGroup(comm_group);
}

bool CommOpDef::DoMapToParallelDevices(const DeviceGroup& placement_group) {
  auto local_device = GetLocalDevice();
  auto& input_op = _inputs[0]->producer();
  if (is_comm_op(input_op)) {
    HT_LOG_DEBUG << local_device << ": " << name() << ": replace input from " << _inputs[0] << " to " << input_op->input(0);     
    ReplaceInput(0, input_op->input(0));
    HT_LOG_DEBUG << local_device << ": " << name() << ": inputs: " << _inputs << ", outputs: " << _outputs;
    // update comm_op type
    get_comm_type();
  }
  return OperatorDef::DoMapToParallelDevices(placement_group);  
}

HTShapeList CommOpDef::DoInferShape(const HTShapeList& input_shapes) {
  const HTShape& input_shape = input_shapes.at(0);

  Tensor& input = _inputs[0];
  auto src_ds = input->get_distributed_states();
  auto dst_ds = get_dst_distributed_states();

  HTShape shape; shape.reserve(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); d++) {
    shape[d] = input_shape[d] * src_ds.get_dim(d) / dst_ds.get_dim(d);
  }
  
  return {shape};
}

TensorList CommOpDef::DoGradient(const TensorList& grad_outputs) {
  Tensor& input = _inputs[0];
  auto ds_input = input->get_distributed_states();
  Tensor& output = _outputs[0];
  auto ds_output = output->get_distributed_states();
  const Tensor& grad_output = grad_outputs.at(0);
  auto ds_grad_output = grad_output->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_output.is_valid())
           << "distributed states for input and output tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_output.get_device_num())
           << "distributed states for input and output tensor must be matched!";
  // HT_ASSERT() // TODO: check ds_grad_output and ds_output must be same
  DistributedStates ds_grad_input(ds_input);
  if (ds_grad_input.get_dim(-2) > 1) { // partial->duplicate
    std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
    auto res_states = ds_grad_input.combine_states(src2dst);
    auto res_order = ds_grad_input.combine_order(src2dst);
    auto device_num = ds_grad_input.get_device_num();
    ds_grad_input.set_distributed_states({device_num, res_states, res_order});
  }
  Tensor grad_input = CommOp(grad_output, ds_grad_input, OpMeta().set_name("grad_" + name()))->output(0);
  return {grad_input};
}

void CommOpDef::DoDeduceStates() {
  Tensor& input = _inputs[0];
  DistributedStates ds_input = input->get_distributed_states();
  DistributedStates ds_dst = get_dst_distributed_states();
  // TODO: check states/order between src and dst
  HT_ASSERT(ds_input.is_valid() && ds_dst.is_valid())
           << "distributed states for input and dst tensor must be valid!";
  HT_ASSERT(ds_input.get_device_num() == ds_dst.get_device_num())
           << "cannot convert src distributed states to unpaired dst distributed states!";
  Tensor& output = _outputs[0];
  output->set_distributed_states(ds_dst);
}

void P2PSendOpDef::BindRecvOp(const P2PRecvOp& recv_op) {
  HT_ASSERT(_recv_op == nullptr) << "Try to bind P2PRecvOp twice";
  _recv_op = std::make_unique<P2PRecvOp>(recv_op);
}

const P2PRecvOp& P2PSendOpDef::recv_op() const {
  HT_ASSERT(_recv_op != nullptr) << "P2PRecvOp is not bound yet";
  return *_recv_op;
}

P2PRecvOp& P2PSendOpDef::recv_op() {
  HT_ASSERT(_recv_op != nullptr) << "P2PRecvOp is not bound yet";
  return *_recv_op;
}

void P2PRecvOpDef::BindSendOp(const P2PSendOp& send_op) {
  HT_ASSERT(_send_op == nullptr) << "Try to bind P2PSendOp twice";
  _send_op = std::make_unique<P2PSendOp>(send_op);
}

const P2PSendOp& P2PRecvOpDef::send_op() const {
  HT_ASSERT(_send_op != nullptr) << "P2PSendOp is not bound yet";
  ;
  return *_send_op;
}

P2PSendOp& P2PRecvOpDef::send_op() {
  HT_ASSERT(_send_op != nullptr) << "P2PSendOp is not bound yet";
  ;
  return *_send_op;
}

bool AllReduceOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  if (_comm_group.empty()) {
    _comm_group = pg;
  } else {
    for (int i = 0; i < _comm_group.num_devices(); i++) {
      HT_ASSERT(pg.contains(_comm_group.get(i))) 
        << "AllReduceOp: device in comm_group: " << _comm_group.get(i) 
        << " must in device group: " << pg;
    }
  }
  HT_ASSERT(_comm_group.num_devices() >= 2)
    << "AllReduce requires two or more devices. Got " << _comm_group;      
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList AllReduceOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  // TODO: support in-place?
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::AllReduce, inputs.at(0),
                                  outputs.at(0), kSUM, _comm_group, stream()); // _comm_group is a subset of placement_group                              
  return outputs;
}

HTShapeList AllReduceOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

bool P2PSendOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  HT_ASSERT(pg.num_devices() == _dst_group.num_devices())
    << "Currently we require equal data parallelism degree across "
    << "P2P communication. Got " << pg << " vs. " << _dst_group;

  _placement_group = pg;
  return true;
}

bool P2PSendOpDef::DoPlaceToLocalDevice(const Device& placement,
                                        StreamIndex stream_id) {
  _index_in_group = _placement_group.get_index(placement);
  HT_ASSERT(is_distributed_tensor_send_op() || !is_distributed_tensor_send_op() && _dst_group.get(_index_in_group) != placement)
    << "Pipeline p2p send op: source and destination are the same (" << placement << ")";
  if (!is_distributed_tensor_send_op()) {
    _dst_device_index = _index_in_group;
  }  
  return OperatorDef::DoPlaceToLocalDevice(placement, stream_id);
}

NDArrayList P2PSendOpDef::DoCompute(const NDArrayList& inputs,
                                    RuntimeContext& ctx) {
  NDArray input = inputs.at(0);
  HT_ASSERT(input->dtype() == _inputs[0]->dtype())
    << "Data type mismatched for P2P communication: " << input->dtype()
    << " vs. " << _inputs[0]->dtype();

  // TODO: sending the shape in compute fn is just a walkaround,
  // we shall determine the shape for recv op in executor
  NDArray send_shape = NDArray::empty({HT_MAX_NDIM + 1}, Device(kCPU), kInt64);
  auto* ptr = send_shape->data_ptr<int64_t>();
  ptr[0] = static_cast<int64_t>(input->ndim());
  std::copy(input->shape().begin(), input->shape().end(), ptr + 1);
  hetu::impl::P2PSendCpu(send_shape, _dst_group.get(_dst_device_index),
                         Stream(Device(kCPU), kBlockingStream));

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::P2PSend, input,
                                  _dst_group.get(_dst_device_index), stream());
  return NDArrayList();
}

bool P2PRecvOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  HT_ASSERT(pg.num_devices() == _src_group.num_devices())
    << "Currently we require equal data parallelism degree across "
    << "P2P communication. Got " << _src_group << " vs. " << pg;

  _placement_group = pg;
  // TODO: set the parallel statuses of output
  for (auto& output : _outputs) {
    output->set_placement_group(pg);
  }  
  return true;
}

bool P2PRecvOpDef::DoPlaceToLocalDevice(const Device& placement,
                                        StreamIndex stream_id) {
  _index_in_group = _placement_group.get_index(placement);
  HT_ASSERT(is_distributed_tensor_recv_op() || !is_distributed_tensor_recv_op() && _src_group.get(_index_in_group) != placement)
    << "Pipeline p2p recv op: source and destination are the same (" << placement << ")";
  if (!is_distributed_tensor_recv_op()) {
    _src_device_index = _index_in_group;
  }
  return OperatorDef::DoPlaceToLocalDevice(placement, stream_id);
}

NDArrayList P2PRecvOpDef::DoCompute(const NDArrayList& inputs,
                                    RuntimeContext& ctx) {
  // TODO: receiving the shape in compute fn is just a walkaround,
  // we shall determine the shape for recv op in executor
  NDArray recv_shape = NDArray::empty({HT_MAX_NDIM + 1}, Device(kCPU), kInt64);
  hetu::impl::P2PRecvCpu(recv_shape, _src_group.get(_src_device_index),
                         Stream(Device(kCPU), kBlockingStream));
  auto* ptr = recv_shape->data_ptr<int64_t>();
  HTShape shape(ptr + 1, ptr + 1 + ptr[0]);
  NDArray output = NDArray::empty(shape, placement(), _outputs[0]->dtype());

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::P2PRecv, output,
                                  _src_group.get(_src_device_index), stream());
  return {output};
}

NDArrayList BatchedISendIRecvOpDef::DoCompute(const NDArrayList& inputs, 
                                              RuntimeContext& ctx) {
  for (int i = 0; i < _inputs.size(); i++) {
    NDArray input = inputs.at(i);
    HT_ASSERT(input->dtype() == _inputs[i]->dtype())
      << "Data type mismatched for ISend communication: " << input->dtype()
      << " vs. " << _inputs[i]->dtype();
  }
  NDArrayList outputs;
  for (int i = 0; i < _outputs.size(); i++) {
    outputs.push_back(NDArray::empty(_outputs[i]->shape(), placement(), _outputs[i]->dtype()));
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), 
                                  hetu::impl::BatchedISendIRecv,
                                  inputs, _dst_devices, outputs, 
                                  _src_devices, _comm_devices, stream());
  return outputs;                                
}

bool AllGatherOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "Allgather: device in comm_group: " << _comm_group.get(i) 
      << " must in device group: " << pg;
  }
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList AllGatherOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  HT_ASSERT(inputs.at(0)->dtype() == _inputs[0]->dtype())
    << "Data type mismatched for AllGather communication: " << inputs.at(0)->dtype()
    << " vs. " << _inputs[0]->dtype();
  NDArray output = NDArray::empty(_outputs[0]->shape(), placement(), _outputs[0]->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::AllGather, inputs.at(0),
                                  output, _comm_group, stream());
  return {output};                                  
}

HTShapeList AllGatherOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape gather_shape = input_shapes.at(0);
  gather_shape[0] *= _comm_group.num_devices();
  return {gather_shape};
}

bool ReduceScatterOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  for (int i = 0; i < _comm_group.num_devices(); i++) {
    HT_ASSERT(pg.contains(_comm_group.get(i))) 
      << "ReduceScatter: device in comm_group: " << _comm_group.get(i) 
      << " must in device group: " << pg;
  }
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList ReduceScatterOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  HT_ASSERT(inputs.at(0)->dtype() == _inputs[0]->dtype())
    << "Data type mismatched for ReduceScatter communication: " << inputs.at(0)->dtype()
    << " vs. " << _inputs[0]->dtype();
  NDArray output = NDArray::empty(_outputs[0]->shape(), placement(), _outputs[0]->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ReduceScatter, inputs.at(0),
                                  output, kSUM, _comm_group, stream());
  return {output};                                  
}

HTShapeList ReduceScatterOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape scatter_shape = input_shapes.at(0);
  scatter_shape[0] /= _comm_group.num_devices();
  HT_ASSERT(scatter_shape[0] >= 1) << "ReduceScatter: input shape[0]: " 
    << input_shapes.at(0)[0] << " must >= comm devices num: " << _comm_group.num_devices();  
  return {scatter_shape};
}

/* BroadcastCommOp */
bool BroadcastCommOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call BroadcastComm with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList BroadcastCommOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  // TODO: support in-place?
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::BroadcastComm, inputs.at(0),
                                  outputs.at(0), broadcaster(), placement_group(), 
                                  stream());
  return outputs;
}

// Reduce type backward
TensorList BroadcastCommOpDef::DoGradient(const TensorList& grad_outputs) {
  return {BroadcastCommGradientOp(grad_outputs.at(0), broadcaster(),
            grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

// // Copy type backward
// TensorList BroadcastCommOpDef::DoGradient(const TensorList& grad_outputs) {
//   return {grad_outputs.at(0)};
// }

HTShapeList BroadcastCommOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

/* BroadcastCommGradientOp */
bool BroadcastCommGradientOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call BroadcastComm with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList BroadcastCommGradientOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ReduceComm, inputs.at(0),
                                  outputs.at(0), broadcaster(), placement_group(), 
                                  stream());
  return outputs;
}

HTShapeList BroadcastCommGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

/* ReduceCommOp */
bool ReduceCommOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call ReduceComm with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList ReduceCommOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  // TODO: support in-place?
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::ReduceComm, inputs.at(0),
                                  outputs.at(0), reducer(), placement_group(), 
                                  stream());
  return outputs;
}

TensorList ReduceCommOpDef::DoGradient(const TensorList& grad_outputs) {
  return {ReduceCommGradientOp(grad_outputs.at(0), reducer(),
            grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList ReduceCommOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

/* ReduceCommGradientOp */
bool ReduceCommGradientOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call ReduceComm with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList ReduceCommGradientOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::BroadcastComm, inputs.at(0),
                                  outputs.at(0), reducer(), placement_group(), 
                                  stream());
  return outputs;
}

HTShapeList ReduceCommGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

// /* GatherOp */
// bool GatherOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
//   // TODO: check whether it satisfies to form a DP group
//   HT_ASSERT(pg.num_devices() >= 2)
//     << "Cannot call ReduceComm with less than 2 devices: " << pg;
//   return OperatorDef::DoMapToParallelDevices(pg);
// }

// NDArrayList GatherOpDef::DoCompute(const NDArrayList& inputs,
//                                       RuntimeContext& ctx) {
//   // TODO: support in-place?
//   NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
//   HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
//                                   hetu::impl::Gather, inputs.at(0),
//                                   outputs.at(0), gatherer(), placement_group(), 
//                                   stream());
//   return outputs;
// }

// TensorList GatherOpDef::DoGradient(const TensorList& grad_outputs) {
//   return {GatherGradientOp(grad_outputs.at(0), gatherer(),
//             grad_op_meta().set_name(grad_name()))
//             ->output(0)};
// }

// HTShapeList GatherOpDef::DoInferShape(const HTShapeList& input_shapes) {
//   HTShape input_shape = input_shapes.at(0);
//   HTShape output_shape(0);
//   output_shape.emplace_back(input_shape[0] * device_group().num_devices());
//   int ndim = input_shape.size();
//   for (int i = 1; i < ndim; ++i) {
//     if (input_shape[i] > 0)
//       output_shape.emplace_back(input_shape[i]);
//   }
//   return {output_shape};
// }

// /* GatherGradientOp */
// bool GatherGradientOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
//   // TODO: check whether it satisfies to form a DP group
//   HT_ASSERT(pg.num_devices() >= 2)
//     << "Cannot call ReduceScatter with less than 2 devices: " << pg;
//   return OperatorDef::DoMapToParallelDevices(pg);
// }

// NDArrayList GatherGradientOpDef::DoCompute(const NDArrayList& inputs,
//                                       RuntimeContext& ctx) {
//   NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
//   HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
//                                   hetu::impl::Scatter, inputs.at(0),
//                                   outputs.at(0), gatherer(), placement_group(), 
//                                   stream());
//   return outputs;
// }

// HTShapeList GatherGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
//   HTShape input_shape = input_shapes.at(0);
//   HTShape output_shape(0);
//   output_shape.emplace_back(input_shape[0] / device_group().num_devices());
//   int ndim = input_shape.size();
//   for (int i = 1; i < ndim; ++i) {
//     if (input_shape[i] > 0)
//       output_shape.emplace_back(input_shape[i]);
//   }
//   return {output_shape};
// }

/* ScatterOp */
bool ScatterOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call ReduceComm with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList ScatterOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  // TODO: support in-place?
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Scatter, inputs.at(0),
                                  outputs.at(0), scatterer(), placement_group(), 
                                  stream());
  return outputs;
}

TensorList ScatterOpDef::DoGradient(const TensorList& grad_outputs) {
  return {ScatterGradientOp(grad_outputs.at(0), scatterer(),
            grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList ScatterOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape input_shape = input_shapes.at(0);
  HTShape output_shape(0);
  output_shape.emplace_back(input_shape[0] / device_group().num_devices());
  int ndim = input_shape.size();
  for (int i = 1; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  return {output_shape};
}

/* ScatterGradientOp */
bool ScatterGradientOpDef::DoMapToParallelDevices(const DeviceGroup& pg) {
  // TODO: check whether it satisfies to form a DP group
  HT_ASSERT(pg.num_devices() >= 2)
    << "Cannot call Scatter with less than 2 devices: " << pg;
  return OperatorDef::DoMapToParallelDevices(pg);
}

NDArrayList ScatterGradientOpDef::DoCompute(const NDArrayList& inputs,
                                      RuntimeContext& ctx) {
  NDArrayList outputs = std::move(DoAllocOutputs(inputs, ctx));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Gather, inputs.at(0),
                                  outputs.at(0), scatterer(), placement_group(), 
                                  stream());
  return outputs;
}

HTShapeList ScatterGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HTShape input_shape = input_shapes.at(0);
  HTShape output_shape(0);
  output_shape.emplace_back(input_shape[0] * device_group().num_devices());
  int ndim = input_shape.size();
  for (int i = 1; i < ndim; ++i) {
    if (input_shape[i] > 0)
      output_shape.emplace_back(input_shape[i]);
  }
  return {output_shape};
}

} // namespace autograd
} // namespace hetu

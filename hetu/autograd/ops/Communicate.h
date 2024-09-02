#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/tensor.h"
#include "hetu/autograd/distributed_states.h"

#include "hetu/impl/communication/comm_group.h"
using namespace hetu::impl::comm;

namespace hetu {
namespace autograd {

class AllReduceOpDef;
class AllReduceOp;
class P2PSendOpDef;
class P2PSendOp;
class P2PRecvOpDef;
class P2PRecvOp;
class CommOpDef;
class CommOp;

class CommOpDef : public OperatorDef {
 private:
  friend class CommOp;
  struct constructor_access_key {};

 public:
  CommOpDef(const constructor_access_key&, Tensor input, 
            DistributedStates dst_distributed_states, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(CommOp), {input}, op_meta), _dst_distributed_states(dst_distributed_states) {
    auto local_device = GetLocalDevice();

    auto src_distributed_states = _inputs[0]->get_distributed_states();
    get_comm_type();

    // 检查src_ds是否valid
    HT_ASSERT(src_distributed_states.is_valid()) 
              << "distributed states for input tensor is not valid!";    
    const HTShape& input_shape = _inputs[0]->shape();
    HTShape shape = input_shape;
    for (int d = 0; d < input_shape.size(); d++) {
      shape[d] = input_shape[d] * src_distributed_states.get_dim(d) / dst_distributed_states.get_dim(d);
    }
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));

    // inputs.ds -> outputs.ds
    DoDeduceStates();
  }

  uint64_t op_indicator() const noexcept {
    return COMM_OP;
  }

  inline DistributedStates get_dst_distributed_states() const {
    return _dst_distributed_states;
  }

  uint64_t get_comm_type();
  DeviceGroup get_devices_by_dim(int32_t dim);
  // DeviceGroup get_allreduce_devices();

  void DoDeduceStates() override;
  
 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override; 
  TensorList DoGradient(const TensorList& grad_outputs) override;
  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;  
  uint64_t _comm_type;
  DistributedStates _dst_distributed_states;
};

class CommOp final : public OpWrapper<CommOpDef> {
 public:
  CommOp(Tensor input, DistributedStates& dst_distributed_states, 
         const OpMeta& op_meta = OpMeta())
  : OpWrapper<CommOpDef>(make_ptr<CommOpDef>(
    CommOpDef::constructor_access_key(), input, dst_distributed_states, op_meta)) {}
};

class AllReduceOpDef : public OperatorDef {
 private:
  friend class AllReduceOp;
  struct constrcutor_access_key {};

 public:
  AllReduceOpDef(const constrcutor_access_key&, Tensor input, 
                 const DeviceGroup& comm_group = DeviceGroup(),
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AllReduceOpDef), {input}, op_meta), _comm_group(comm_group) {
    auto& _device_group = device_group();
    if (!_device_group.empty()) {
      // TODO: check whether it satisfies to form a DP group
      if (comm_group.empty()) {
        _comm_group = _device_group;
      } else {
        for (int i = 0; i < comm_group.num_devices(); i++) {
          HT_ASSERT(_device_group.contains(comm_group.get(i))) 
            << "AllReduceOp: device in comm_group: " << comm_group.get(i) 
            << " must in device group: " << _device_group;
        }
      }
      HT_ASSERT(_comm_group.num_devices() >= 2)
        << "AllReduce requires two or more devices. Got " << _comm_group;
    }
    AddOutput(input->meta());
  }

  uint64_t op_indicator() const noexcept {
    return ALL_REDUCE_OP;
  }

  const DeviceGroup& comm_group() const {
    return _comm_group;
  } 

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  DeviceGroup _comm_group;
};

class AllReduceOp final : public OpWrapper<AllReduceOpDef> {
 public:
  AllReduceOp(Tensor input, const DeviceGroup& comm_group = DeviceGroup(), 
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<AllReduceOpDef>(make_ptr<AllReduceOpDef>(
      AllReduceOpDef::constrcutor_access_key(), input, comm_group, op_meta)) {}
};

class P2PSendOpDef : public OperatorDef {
 private:
  friend class P2PSendOp;
  struct constrcutor_access_key {};

 public:
  P2PSendOpDef(const constrcutor_access_key&, Tensor input,
               const DeviceGroup& dst_group, int dst_device_index = -1, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(P2PSendOp), {input}, op_meta), _dst_group(dst_group), _dst_device_index(dst_device_index) {
    HT_ASSERT(!dst_group.empty())
      << "Please provide the \"dst_group\" argument to indicate "
      << "the destination devices for P2PSend";
    HT_ASSERT(device_group().empty() ||
              device_group().num_devices() == dst_group.num_devices())
      << "Currently we require equal data parallelism degree across "
      << "P2P communication. Got " << device_group() << " vs. " << dst_group;
  }

  // Walkaround: bind the send and recv op
  void BindRecvOp(const P2PRecvOp& recv_op);

  const P2PRecvOp& recv_op() const;

  P2PRecvOp& recv_op();

  int index_in_group() {
    return _index_in_group;
  }

  void set_index_in_group(int index) {
    // 要么是在map to local device的时候已经赋值好了, 要么就是在这里赋值(用于distributed tensor中非local device的p2p op的赋值)
    HT_ASSERT(_index_in_group == -1) << "only allow set when _index_in_group = -1!";
    _index_in_group = index;
  }  

  const DeviceGroup& dst_group() const {
    return _dst_group;
  }

  int dst_device_index() {
    return _dst_device_index;
  }

  bool is_distributed_tensor_send_op() {
    return _dst_device_index != -1;
  }  

  uint64_t op_indicator() const noexcept {
    return PEER_TO_PEER_SEND_OP;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  DeviceGroup _dst_group;
  int _dst_device_index{-1}; // for distributed tensor p2p
  int _index_in_group{-1}; // for pipeline p2p
  // The `_recv_op` member is wrapped into a unique pointer since
  // the forward declared `P2PRecvOp` class has incomplete type now.
  std::unique_ptr<P2PRecvOp> _recv_op;
};

class P2PSendOp final : public OpWrapper<P2PSendOpDef> {
 public:
  P2PSendOp(Tensor input, const DeviceGroup& dst_group, 
            int dst_device_index = -1, const OpMeta& op_meta = OpMeta())
  : OpWrapper<P2PSendOpDef>(make_ptr<P2PSendOpDef>(
      P2PSendOpDef::constrcutor_access_key(), input, dst_group, dst_device_index, op_meta)) {}
};

class P2PRecvOpDef : public OperatorDef {
 private:
  friend class P2PRecvOp;
  struct constrcutor_access_key {};

 public:
  P2PRecvOpDef(const constrcutor_access_key&, const DeviceGroup& src_group, 
               DataType dtype, const HTShape& shape = HTShape(), int src_device_index = -1,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(P2PRecvOp), TensorList(), op_meta),
    _src_group(src_group), _src_device_index(src_device_index) {
    HT_ASSERT(!src_group.empty())
      << "Please provide the \"src_group\" argument to indicate "
      << "the source devices for P2PRecv";
    HT_ASSERT(device_group().empty() ||
              src_group.num_devices() == device_group().num_devices())
      << "Currently we require equal data parallelism degree across "
      << "P2P communication. Got " << src_group << " vs. " << device_group();
    HT_ASSERT(dtype != kUndeterminedDataType)
      << "Please specify data type for P2P communication";
    AddOutput(NDArrayMeta().set_dtype(dtype).set_shape(shape));
  }

  // Walkaround: bind the send and recv op
  void BindSendOp(const P2PSendOp& send_op);

  const P2PSendOp& send_op() const;

  P2PSendOp& send_op();

  int index_in_group() {
    return _index_in_group;
  }

  void set_index_in_group(int index) {
    // 要么是在map to local device的时候已经赋值好了, 要么就是在这里赋值(用于distributed tensor中非local device的p2p op的赋值)
    HT_ASSERT(_index_in_group == -1) << "only allow set when _index_in_group = -1!";
    _index_in_group = index;
  }

  const DeviceGroup& src_group() const {
    return _src_group;
  }

  int src_device_index() {
    return _src_device_index;
  }

  bool is_distributed_tensor_recv_op() {
    return _src_device_index != -1;
  }

  uint64_t op_indicator() const noexcept {
    return PEER_TO_PEER_RECV_OP;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  OpList _linked_ops{};
  DeviceGroup _src_group;
  int _src_device_index{-1}; // for distributed tensor p2p
  int _index_in_group{-1}; // for pipeline p2p
  std::unique_ptr<P2PSendOp> _send_op;
};

class P2PRecvOp final : public OpWrapper<P2PRecvOpDef> {
 public:
  P2PRecvOp(const DeviceGroup& src_group, DataType dtype,
            const HTShape& shape = HTShape(), int src_device_index = -1, 
            const OpMeta& op_meta = OpMeta())
  : OpWrapper<P2PRecvOpDef>(
      make_ptr<P2PRecvOpDef>(P2PRecvOpDef::constrcutor_access_key(), src_group,
                             dtype, shape, src_device_index, op_meta)) {}
};

class BatchedISendIRecvOpDef : public OperatorDef {
 private:
  friend class BatchedISendIRecvOp;
  struct constrcutor_access_key {};
  
 public:
  BatchedISendIRecvOpDef(const constrcutor_access_key&, TensorList& inputs, 
                         std::vector<Device>& dst_devices, HTShapeList& outputs_shape, 
                         std::vector<Device>& src_devices, std::vector<Device>& comm_devices,
                         DataType dtype, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BatchedISendIRecvOp), inputs, op_meta), _src_devices(src_devices), 
                _dst_devices(dst_devices), _comm_devices(comm_devices) {
    HT_ASSERT((inputs.size() == dst_devices.size()) && (outputs_shape.size() == src_devices.size())) 
      << "Send/Recv data must be matched with dst/src Devices!";
    print_mesg();
    std::vector<NDArrayMeta> output_meta_list;
    for (auto& output_shape : outputs_shape) {
      output_meta_list.push_back(NDArrayMeta().set_dtype(dtype).set_shape(output_shape));
    }
    AddOutputs(output_meta_list);
  }

  void print_mesg() {
    std::string dst = "dst devices =";
    for (auto& d : _dst_devices) {
      dst += " device_" + std::to_string(d.index());
    }
    std::string src = "src devices =";
    for (auto& s : _src_devices) {
      src += " device_" + std::to_string(s.index());
    }
    auto local_device = GetLocalDevice();
    HT_LOG_DEBUG << local_device << ": BatchedISendIRecvOp definition: " << name() << ": " << dst << ", " << src;    
  }

  std::vector<Device> src_devices() {
    return _src_devices;
  }

  std::vector<Device> dst_devices() {
    return _dst_devices;
  }

  uint64_t op_indicator() const noexcept {
    return BATCHED_ISEND_IRECV_OP;
  }

 protected:
  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;  

  std::vector<Device> _dst_devices;
  std::vector<Device> _src_devices;
  std::vector<Device> _comm_devices;
};

class BatchedISendIRecvOp final : public OpWrapper<BatchedISendIRecvOpDef> {
 public:
  BatchedISendIRecvOp(TensorList& inputs, std::vector<Device>& dst_devices,
                      HTShapeList& outputs_shape, std::vector<Device>& src_devices, 
                      std::vector<Device>& comm_devices, DataType dtype, 
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<BatchedISendIRecvOpDef>(
    make_ptr<BatchedISendIRecvOpDef>(BatchedISendIRecvOpDef::constrcutor_access_key(), 
    inputs, dst_devices, outputs_shape, src_devices, comm_devices, dtype, op_meta)) {}                      
};

class AllGatherOpDef : public OperatorDef {
 private:
  friend class AllGatherOp;
  struct constructor_access_key {};
 public:
  AllGatherOpDef(const constructor_access_key&, Tensor input,
                 const DeviceGroup& comm_group, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AllGatherOpDef), {input}, op_meta), _comm_group(comm_group) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
      << "AllGather requires two or more devices. Got " << _comm_group;          
    auto& _device_group = device_group();
    if (!_device_group.empty()) {
      for (int i = 0; i < comm_group.num_devices(); i++) {
        HT_ASSERT(_device_group.contains(comm_group.get(i)))
          << "AllGather: device in comm_group: " << comm_group.get(i) 
          << " must in device group: " << _device_group;
      }
    }
    DataType dtype = input->dtype();
    HTShape gather_shape = input->shape();
    gather_shape[0] *= _comm_group.num_devices();
    AddOutput(NDArrayMeta().set_dtype(dtype).set_shape(gather_shape));
  }                 
 
 uint64_t op_indicator() const noexcept {
  return ALL_GATHER_OP;
 }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  DeviceGroup _comm_group;
};

class AllGatherOp final : public OpWrapper<AllGatherOpDef> {
 public:
  AllGatherOp(Tensor input, const DeviceGroup& comm_group,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<AllGatherOpDef>(make_ptr<AllGatherOpDef>(
      AllGatherOpDef::constructor_access_key(), input, comm_group, op_meta)) {}              
};

class ReduceScatterOpDef : public OperatorDef {
 private:
  friend class ReduceScatterOp;
  struct constructor_access_key {};
 public:
  ReduceScatterOpDef(const constructor_access_key&, Tensor input,
                     const DeviceGroup& comm_group, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceScatterOpDef), {input}, op_meta), _comm_group(comm_group) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
      << "ReduceScatter requires two or more devices. Got " << _comm_group;          
    auto& _device_group = device_group();
    if (!_device_group.empty()) {
      for (int i = 0; i < comm_group.num_devices(); i++) {
        HT_ASSERT(_device_group.contains(comm_group.get(i)))
          << "ReduceScatter: device in comm_group: " << comm_group.get(i) 
          << " must in device group: " << _device_group;
      }
    }
    DataType dtype = input->dtype();
    HTShape scatter_shape = input->shape();
    scatter_shape[0] /= _comm_group.num_devices();
    HT_ASSERT(scatter_shape[0] >= 1) << "ReduceScatter: input shape[0]: " 
      << input->shape()[0] << " must >= comm devices num: " << _comm_group.num_devices();
    AddOutput(NDArrayMeta().set_dtype(dtype).set_shape(scatter_shape));
  }                 
 
 uint64_t op_indicator() const noexcept {
  return REDUCE_SCATTER_OP;
 }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  DeviceGroup _comm_group;
};

class ReduceScatterOp final : public OpWrapper<ReduceScatterOpDef> {
 public:
  ReduceScatterOp(Tensor input, const DeviceGroup& comm_group,
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceScatterOpDef>(make_ptr<ReduceScatterOpDef>(
      ReduceScatterOpDef::constructor_access_key(), input, comm_group, op_meta)) {}              
};

/* BroadcastCommOp */
class BroadcastCommOpDef : public OperatorDef {
 private:
  friend class BroadcastCommOp;
  struct constrcutor_access_key {};

 public:
  BroadcastCommOpDef(const constrcutor_access_key&, Tensor input, int broadcaster,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastCommOpDef), {input}, op_meta),
    _broadcaster(broadcaster) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << "BroadcastComm requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  uint64_t op_indicator() const noexcept {
    return BROADCAST_OP;
  }

  int broadcaster() const {
    return _broadcaster;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  int _broadcaster;
};

class BroadcastCommOp final : public OpWrapper<BroadcastCommOpDef> {
 public:
  BroadcastCommOp(Tensor input, int broadcaster, const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastCommOpDef>(make_ptr<BroadcastCommOpDef>(
     BroadcastCommOpDef::constrcutor_access_key(), input, broadcaster, op_meta)) {}
};

/* BroadcastCommGradientOp */
class BroadcastCommGradientOpDef : public OperatorDef {
 private:
  friend class BroadcastCommGradientOp;
  struct constrcutor_access_key {};

 public:
  BroadcastCommGradientOpDef(const constrcutor_access_key&, Tensor input, int broadcaster,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(BroadcastCommGradientOpDef), {input}, op_meta),
    _broadcaster(broadcaster) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << "BroadcastComm requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  int broadcaster() const {
    return _broadcaster;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int _broadcaster;
};

class BroadcastCommGradientOp final : public OpWrapper<BroadcastCommGradientOpDef> {
 public:
  BroadcastCommGradientOp(Tensor input, int broadcaster, const OpMeta& op_meta = OpMeta())
  : OpWrapper<BroadcastCommGradientOpDef>(make_ptr<BroadcastCommGradientOpDef>(
     BroadcastCommGradientOpDef::constrcutor_access_key(), input, broadcaster, op_meta)) {}
};

/* ReduceCommOp */
class ReduceCommOpDef : public OperatorDef {
 private:
  friend class ReduceCommOp;
  struct constrcutor_access_key {};

 public:
  ReduceCommOpDef(const constrcutor_access_key&, Tensor input, int reducer,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceCommOpDef), {input}, op_meta),
    _reducer(reducer) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << "ReduceComm requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  uint64_t op_indicator() const noexcept {
    return REDUCE_OP;
  }

  int reducer() const {
    return _reducer;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;  

  int _reducer;
};

class ReduceCommOp final : public OpWrapper<ReduceCommOpDef> {
 public:
  ReduceCommOp(Tensor input, int reducer, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceCommOpDef>(make_ptr<ReduceCommOpDef>(
      ReduceCommOpDef::constrcutor_access_key(), input, reducer, op_meta)) {}
};

/* ReduceCommGradientOp */
class ReduceCommGradientOpDef : public OperatorDef {
 private:
  friend class ReduceCommGradientOp;
  struct constrcutor_access_key {};

 public:
  ReduceCommGradientOpDef(const constrcutor_access_key&, Tensor input, int reducer,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceCommGradientOpDef), {input}, op_meta),
    _reducer(reducer) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << "ReduceComm requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  int reducer() const {
    return _reducer;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int _reducer;
};

class ReduceCommGradientOp final : public OpWrapper<ReduceCommGradientOpDef> {
 public:
  ReduceCommGradientOp(Tensor input, int reducer, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceCommGradientOpDef>(make_ptr<ReduceCommGradientOpDef>(
      ReduceCommGradientOpDef::constrcutor_access_key(), input, reducer, op_meta)) {}
};

// /* GatherOp */
// class GatherOpDef : public OperatorDef {
//  private:
//   friend class GatherOp;
//   struct constrcutor_access_key {};

//  public:
//    GatherOpDef(const constrcutor_access_key&, Tensor input, int gatherer,
//                  const OpMeta& op_meta = OpMeta())
//   : OperatorDef(quote(GatherOpDef), {input}, op_meta),
//     _gatherer(gatherer) {
//     if (!device_group().empty()) {
//       // TODO: check whether it satisfies to form a DP group
//       HT_ASSERT(device_group().num_devices() >= 2)
//         << " Gather requires two or more devices. Got " << device_group();
//     }
//     AddOutput(input->meta());
//   }

//   uint64_t op_indicator() const noexcept {
//     return GATHER_OP;
//   }

//   int gatherer() const {
//     return _gatherer;
//   }

//  protected:
//   bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

//   NDArrayList DoCompute(const NDArrayList& inputs,
//                         RuntimeContext& ctx) override;

//   HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

//   TensorList DoGradient(const TensorList& grad_outputs) override;  

//   int _gatherer;
// };

// class GatherOp final : public OpWrapper<GatherOpDef> {
//  public:
//    GatherOp(Tensor input, int gatherer, const OpMeta& op_meta = OpMeta())
//   : OpWrapper<GatherOpDef>(make_ptr<GatherOpDef>(
//        GatherOpDef::constrcutor_access_key(), input, gatherer, op_meta)) {}
// };

// /* GatherGradientOp */
// class GatherGradientOpDef : public OperatorDef {
//  private:
//   friend class GatherGradientOp;
//   struct constrcutor_access_key {};

//  public:
//    GatherGradientOpDef(const constrcutor_access_key&, Tensor input, int gatherer,
//                  const OpMeta& op_meta = OpMeta())
//   : OperatorDef(quote(GatherGradientOpDef), {input}, op_meta),
//     _gatherer(gatherer) {
//     if (!device_group().empty()) {
//       // TODO: check whether it satisfies to form a DP group
//       HT_ASSERT(device_group().num_devices() >= 2)
//         << " Gather requires two or more devices. Got " << device_group();
//     }
//     AddOutput(input->meta());
//   }

//   int gatherer() const {
//     return _gatherer;
//   }

//  protected:
//   bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

//   NDArrayList DoCompute(const NDArrayList& inputs,
//                         RuntimeContext& ctx) override;

//   HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

//   int _gatherer;
// };

// class GatherGradientOp final : public OpWrapper<GatherGradientOpDef> {
//  public:
//    GatherGradientOp(Tensor input, int gatherer, const OpMeta& op_meta = OpMeta())
//   : OpWrapper<GatherGradientOpDef>(make_ptr<GatherGradientOpDef>(
//        GatherGradientOpDef::constrcutor_access_key(), input, gatherer, op_meta)) {}
// };

/* ScatterOp */
class ScatterOpDef : public OperatorDef {
 private:
  friend class ScatterOp;
  struct constrcutor_access_key {};

 public:
   ScatterOpDef(const constrcutor_access_key&, Tensor input, int scatterer,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ScatterOpDef), {input}, op_meta),
    _scatterer(scatterer) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << " Scatter requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  uint64_t op_indicator() const noexcept {
    return SCATTER_OP;
  }

  int scatterer() const {
    return _scatterer;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  int _scatterer;
};

class ScatterOp final : public OpWrapper<ScatterOpDef> {
 public:
   ScatterOp(Tensor input, int scatterer, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ScatterOpDef>(make_ptr<ScatterOpDef>(
       ScatterOpDef::constrcutor_access_key(), input, scatterer, op_meta)) {}
};

/* ScatterGradientOp */
class ScatterGradientOpDef : public OperatorDef {
 private:
  friend class ScatterGradientOp;
  struct constrcutor_access_key {};

 public:
   ScatterGradientOpDef(const constrcutor_access_key&, Tensor input, int scatterer,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ScatterGradientOpDef), {input}, op_meta),
    _scatterer(scatterer) {
    if (!device_group().empty()) {
      // TODO: check whether it satisfies to form a DP group
      HT_ASSERT(device_group().num_devices() >= 2)
        << " Scatter requires two or more devices. Got " << device_group();
    }
    AddOutput(input->meta());
  }

  int scatterer() const {
    return _scatterer;
  }

 protected:
  bool DoMapToParallelDevices(const DeviceGroup& placement_group) override;

  NDArrayList DoCompute(const NDArrayList& inputs,
                        RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int _scatterer;
};

class ScatterGradientOp final : public OpWrapper<ScatterGradientOpDef> {
 public:
   ScatterGradientOp(Tensor input, int scatterer, const OpMeta& op_meta = OpMeta())
  : OpWrapper<ScatterGradientOpDef>(make_ptr<ScatterGradientOpDef>(
       ScatterGradientOpDef::constrcutor_access_key(), input, scatterer, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

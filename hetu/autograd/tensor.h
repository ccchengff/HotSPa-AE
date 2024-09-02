#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/autograd/common.h"
#include "hetu/autograd/op_meta.h"
#include "hetu/autograd/init/initializer.h"
#include "hetu/autograd/distributed_states.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include <functional>

namespace hetu {
namespace autograd {

/******************************************************
 * Tensor Definition
 ******************************************************/

class Tensor : public shared_ptr_wrapper<TensorDef> {
 public:
  Tensor() = default;
  Tensor(const TensorName& name, int32_t output_id,
         const NDArrayMeta& meta = {}, const DistributedStates& distributed_states = {});
};

class TensorDef : public shared_ptr_target {
 protected:
  friend class OperatorDef;
  template <typename T>
  friend class OpWrapper;
  friend class Tensor;
  struct constrcutor_access_key {};

 public:
  TensorDef(const constrcutor_access_key&, const TensorName& name,
            int32_t output_id, const NDArrayMeta& meta = {}, const DistributedStates& distributed_states = {})
  : _id{_next_tensor_id()}, _name(name), _output_id(output_id), _meta(meta), _distributed_states(distributed_states) {}

  ~TensorDef() = default;

  // disable copy constructor and move constructor
  TensorDef(const TensorDef&) = delete;
  TensorDef& operator=(const TensorDef&) = delete;
  TensorDef(TensorDef&&) = delete;
  TensorDef& operator=(TensorDef&&) = delete;

  TensorId id() const {
    return _id;
  }

  TensorName name() const {
    return _name;
  }

  const Operator& producer() const;

  Operator& producer();

  int32_t output_id() const noexcept {
    return _output_id;
  }

  bool is_tensor() const noexcept {
    return _output_id >= 0;
  }

  size_t num_consumers() const;

  const Operator& consumer(size_t i) const;

  Operator& consumer(size_t i);

  const NDArrayMeta& meta() const noexcept {
    return _meta;
  }

  size_t ndim() const {
    return _meta.ndim();
  }

  size_t numel() const {
    return _meta.numel();
  }

  DataType dtype() const {
    return _meta.dtype;
  }

  const Device& device() const noexcept {
    return _meta.device;
  }

  bool is_cpu() const {
    return _meta.device.is_cpu();
  }

  bool is_cuda() const {
    return _meta.device.is_cuda();
  }

  const HTShape& shape() const {
    return _meta.shape;
  }

  int64_t shape(size_t axis) const {
    return _meta.shape[axis];
  }

  const HTStride& stride() const {
    return _meta.stride;
  }

  int64_t stride(size_t axis) const {
    return _meta.stride[axis];
  }

  bool has_shape() const {
    return _meta.shape.size() > 0;
  }

  const Device& placement() const noexcept {
    return device();
  }

  void set_placement(const Device& p) {
    _meta.set_device(p);
    _distributed_states.set_placement(p);
  }

  bool is_computed() const {
    return _computed;
  }

  NDArray& GetOrCompute();

  bool is_variable() const;

  Tensor to_variable(bool trainable = false, const OpMeta& op_meta = OpMeta());

  bool is_trainable() const;

  void set_trainable(bool trainable);

  void reset_initializer(const Initializer& init);

  void reset_data(const NDArray& data);

  Tensor& Gradient();

  const Tensor& Gradient() const;

  void Backward(const Tensor& grad = Tensor());

  void AccumulateGrad(const Tensor& grad);

  void ZeroGrad();

  DistributedStates get_distributed_states() {
    return _distributed_states;
  }

  void set_distributed_states(const DistributedStates& distributed_states) {
    _distributed_states.set_distributed_states(distributed_states);
  }

  // do when MapToParallelDevices
  void set_placement_group(const DeviceGroup& placement_group) {
    _distributed_states.set_placement_group(placement_group);
  }

 protected:
  void SetProducer(Operator& op);

  void AddConsumer(Operator& op);

  // Walkaround methods to get the corresponding wrapper
  Tensor& get_self();

  const Tensor& get_self() const;

  const TensorId _id;
  const TensorName _name;
  const int32_t _output_id;
  NDArrayMeta _meta;

  // The `_producer` member is wrapped into a unique pointer since
  // the forward declared `Operator` class has incomplete type now.
  std::unique_ptr<Operator> _producer;
  OpList _consumers;

  // for define-by-run mode
  bool _computed{false};
  NDArray _data;
  Tensor _grad;

  // for distributed attributes
  DistributedStates _distributed_states;  

 private:
  static TensorId _next_tensor_id() {
    static std::atomic<TensorId> _global_tensor_id(0);
    return _global_tensor_id++;
  }
};

/******************************************************
 * Logging & Streaming
 ******************************************************/

std::ostream& operator<<(std::ostream&, const Tensor&);

} // namespace autograd
} // namespace hetu

namespace std {
inline std::string to_string(const hetu::autograd::Tensor& tensor) {
  std::ostringstream os;
  os << tensor;
  return os.str();
}
} // namespace std

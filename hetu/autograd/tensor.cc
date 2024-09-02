#include "hetu/autograd/tensor.h"
#include "hetu/execution/dbr_executor.h"
#include "hetu/autograd/autograd.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/distributed_states.h"
#include "queue"

namespace hetu {
namespace autograd {

Tensor::Tensor(const TensorName& name, int32_t output_id,
               const NDArrayMeta& meta, const DistributedStates& distributed_states)
: shared_ptr_wrapper<TensorDef>() {
  _ptr = make_ptr<TensorDef>(TensorDef::constrcutor_access_key(), name,
                             output_id, meta, distributed_states);
}

NDArray& TensorDef::GetOrCompute() {
  if (!is_computed()) {
    auto& exec = hetu::execution::GetOrCreateDBRExecutor();
    if (is_tensor())
      exec.Run({producer()->output(output_id())});
    else
      exec.Run({producer()->out_dep_linker()});
    HT_ASSERT(is_computed()) << "Tensor " << name() << " is not properly "
                             << "marked as computed.";
  }
  return _data;
}

bool TensorDef::is_variable() const {
  return is_variable_op(producer()) && is_tensor();
}

Tensor TensorDef::to_variable(bool trainable, const OpMeta& op_meta) {
  HT_ASSERT(is_tensor()) << "Cannot detach dependency linker as a variable";
  if (is_variable()) {
    set_trainable(trainable);
    return get_self();
  } else {
    return VariableOp(GetOrCompute(), trainable,
                      OpMeta::Merge(producer()->op_meta(), op_meta))
      ->output(0);
  }
}

bool TensorDef::is_trainable() const {
  return is_trainable_op(producer()) && is_tensor();
}

void TensorDef::set_trainable(bool trainable) {
  HT_ASSERT(is_variable()) << "Cannot set non-variable tensors as trainable";
  reinterpret_cast<VariableOp&>(producer())->set_trainable(trainable);
}

void TensorDef::reset_initializer(const Initializer& init) {
  HT_ASSERT(is_variable())
    << "Cannot reset initializers for non-variable tensors";
  reinterpret_cast<VariableOp&>(producer())->reset_initializer(init);
}

void TensorDef::reset_data(const NDArray& data) {
  HT_ASSERT(is_variable()) << "Cannot reset data for non-variable tensors";
  reinterpret_cast<VariableOp&>(producer())->reset_data(data);
}

Tensor& TensorDef::Gradient() {
  return _grad;
}

const Tensor& TensorDef::Gradient() const {
  return _grad;
}

void TensorDef::AccumulateGrad(const Tensor& grad) {
  if (!is_trainable()) {
    HT_LOG_WARN << "Trying to update a non-trainable variable. Will ignore it";
    return;
  }

  HT_ASSERT(grad->shape() == shape())
    << "Gradient shape " << grad->shape() << " does not match variable shape "
    << shape() << name();
  if (_grad.is_defined()) {
    // // TODO: add in place
    // _grad = AddElewiseOp(_grad, grad)->output(0);
    HT_NOT_IMPLEMENTED << "Gradient accumulation not implemented";
  } else {
    _grad = grad;
  }
}

void TensorDef::ZeroGrad() {
  // Question: support set as zeros?
  if (_grad.is_defined())
    _grad = Tensor();
}

void TensorDef::Backward(const Tensor& grad) {
  // Question: should we forbid calling `Backward` twice? If yes, then how?
  HT_ASSERT(is_tensor()) << "Cannot call \"Backward\" for a non-Tensor output";
  const auto& producer_op = producer();
  const auto& self = producer()->output(output_id());
  auto topo_order = TopoSort(producer_op);
  TensorList vars;
  vars.reserve(topo_order.size());
  for (auto& op : topo_order)
    if (is_trainable_op(op))
      vars.push_back(op->output(0));
  TensorList grads = Gradients(self, vars, grad);
  HT_ASSERT_EQ(grads.size(), vars.size())
    << "Only " << grads.size() << " gradients are returned for " << vars.size()
    << " variables.";
//   HT_LOG_INFO << grads.size() << " " << name();
  for (size_t i = 0; i < grads.size(); i++) {
    vars[i]->AccumulateGrad(grads[i]);
    // HT_LOG_INFO << vars[i]->name() << " " << grads[i]->GetOrCompute();
  }
}

Tensor& TensorDef::get_self() {
  if (is_tensor())
    return producer()->output(output_id());
  else
    return producer()->out_dep_linker();
}

const Tensor& TensorDef::get_self() const {
  if (is_tensor())
    return producer()->output(output_id());
  else
    return producer()->out_dep_linker();
}

void TensorDef::SetProducer(Operator& op) {
  HT_ASSERT(_producer == nullptr) << "Try to set the producer twice";
  _producer = std::make_unique<Operator>(op);
}

void TensorDef::AddConsumer(Operator& op) {
  _consumers.push_back(op);
}

const Operator& TensorDef::producer() const {
  HT_ASSERT(_producer != nullptr && (*_producer).is_defined())
    << "Producer is not set yet";
  return *_producer;
}

Operator& TensorDef::producer() {
  HT_ASSERT(_producer != nullptr && (*_producer).is_defined())
    << "Producer is not set yet";
  return *_producer;
}

size_t TensorDef::num_consumers() const {
  return _consumers.size();
}

const Operator& TensorDef::consumer(size_t i) const {
  return _consumers[i];
}

Operator& TensorDef::consumer(size_t i) {
  return _consumers[i];
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  if (tensor.is_defined())
    os << tensor->name();
  else
    os << "Tensor()";
  return os;
}

} // namespace autograd
} // namespace hetu

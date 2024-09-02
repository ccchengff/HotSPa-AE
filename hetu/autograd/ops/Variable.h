#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/init/initializer.h"

namespace hetu {
namespace autograd {

namespace {
inline DataType _InferDataType(const NDArray& data, DataType dtype) {
  return data.is_defined() ? data->dtype() : dtype;
}
inline HTShape _InferShape(const NDArray& data, const HTShape& shape) {
  return data.is_defined() ? data->shape() : shape;
}
} // namespace

class PlaceholderOpDef;
class PlaceholderOp;
class VariableOpDef;
class VariableOp;

class PlaceholderBaseOpDef : public OperatorDef {
 protected:
  PlaceholderBaseOpDef(OpType&& op_type, const NDArray& data,
                       std::shared_ptr<Initializer> init, const HTShape shape,
                       bool trainable, DataType dtype, const OpMeta& op_meta)
  : OperatorDef(std::move(op_type), TensorList(), op_meta),
    _data(data),
    _init(init),
    _trainable(trainable),
    _shape(_InferShape(data, shape)) {
    auto dtype_ = _InferDataType(data, dtype);
    HT_ASSERT_NE(dtype_, kUndeterminedDataType)
      << "Failed to construct the \"" << type() << "\" operation "
      << "(with name \"" << name() << "\"): "
      << "Data type is not prodived and cannot be inferred.";
    if (!_data.is_defined()) {
      HT_ASSERT(!_trainable || init != nullptr)
        << "Please provide data or initializers for trainable variables";
      HT_ASSERT(init == nullptr || NumEl(_shape) > 0)
        << "Please provide shape for variables";
      AddOutput(NDArrayMeta().set_dtype(dtype_).set_shape(_shape));
    } else {
      AddOutput(_data->meta());
    }
  }

 public:
  bool need_feed() const {
    return !_data.is_defined() && _init == nullptr;
  }

  const NDArray& data() const {
    return _data;
  }

  const Initializer& initializer() const {
    return *_init;
  }

  const HTShape& shape() const {
    return _shape;
  }

  bool trainable() const {
    return _trainable;
  }

  uint64_t op_indicator() const noexcept {
    return PLACEHOLDER_OP;
  }

 protected:
  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  NDArray _data;
  std::shared_ptr<Initializer> _init;
  bool _trainable;
  HTShape _shape;
};

class PlaceholderOpDef : public PlaceholderBaseOpDef {
 private:
  friend class PlaceholderOp;
  struct constrcutor_access_key {};

 public:
  PlaceholderOpDef(const constrcutor_access_key&, DataType dtype = kFloat32,
                   const HTShape& shape = {}, const OpMeta& op_meta = OpMeta())
  : PlaceholderBaseOpDef(quote(PlaceholderOp), NDArray(), nullptr, shape, false,
                         dtype, op_meta) {}

  uint64_t op_indicator() const noexcept {
    return PLACEHOLDER_OP;
  }
};

class PlaceholderOp final : public OpWrapper<PlaceholderOpDef> {
 public:
  PlaceholderOp(DataType dtype = kFloat32, const HTShape& shape = {},
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<PlaceholderOpDef>(make_ptr<PlaceholderOpDef>(
      PlaceholderOpDef::constrcutor_access_key(), dtype, shape, op_meta)) {}
};

class Optimizer;

class VariableOpDef : public PlaceholderBaseOpDef {
 private:
  friend class Optimizer;
  friend class VariableOp;
  int _inited;
  struct constrcutor_access_key {};

 public:
  VariableOpDef(const constrcutor_access_key&, const NDArray& data,
                bool trainable = false, const OpMeta& op_meta = OpMeta())
  : PlaceholderBaseOpDef(quote(VariableOp), data, nullptr, data->shape(),
                         trainable, data->dtype(), op_meta), _inited(0) {}

  VariableOpDef(const constrcutor_access_key&, const HTShape& shape,
                const Initializer& init, DataType dtype = kFloat32,
                bool trainable = false, const OpMeta& op_meta = OpMeta())
  : PlaceholderBaseOpDef(quote(VariableOp), NDArray(),
                         std::shared_ptr<Initializer>(init.copy()), shape,
                         trainable, dtype, op_meta), _inited(0) {}

  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  void reset_initializer(const Initializer& init);

  void reset_data(const NDArray& data);

  void set_trainable(bool trainable) {
    _trainable = trainable;
  }

  uint64_t op_indicator() const noexcept {
    return VARIABLE_OP;
  }
};

class VariableOp final : public OpWrapper<VariableOpDef> {
 public:
  VariableOp(const NDArray& data, bool trainable = false,
             const OpMeta& op_meta = OpMeta())
  : OpWrapper<VariableOpDef>(make_ptr<VariableOpDef>(
      VariableOpDef::constrcutor_access_key(), data, trainable, op_meta)) {}

  VariableOp(const HTShape& shape, const Initializer& init,
             DataType dtype = kFloat32, bool trainable = false,
             const OpMeta& op_meta = OpMeta())
  : OpWrapper<VariableOpDef>(
      make_ptr<VariableOpDef>(VariableOpDef::constrcutor_access_key(), shape,
                              init, dtype, trainable, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

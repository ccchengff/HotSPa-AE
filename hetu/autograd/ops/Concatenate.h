#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class ConcatenateOpDef;
class ConcatenateOp;
class ConcatenateGradientOpDef;
class ConcatenateGradientOp;

class ConcatenateOpDef : public OperatorDef {
 private:
  friend class ConcatenateOp;
  struct constrcutor_access_key {};

 public:
  ConcatenateOpDef(const constrcutor_access_key&, const TensorList& inputs,
                   size_t axis = 0, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ConcatenateOp), inputs, op_meta), _axis(axis) {
    DoInferMeta();
    if (op_meta.is_deduce_states) {
      DoDeduceStates();
    }
  }

  size_t get_axis() const {
    return _axis;
  }

  int64_t get_grad_offset(size_t idx) const {
    return grad_offsets[idx];
  }

  size_t grad_num() const {
    return grad_offsets.size();
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;  

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _axis;

  std::vector<int64_t> grad_offsets;
};

class ConcatenateOp final : public OpWrapper<ConcatenateOpDef> {
 public:
  ConcatenateOp(const TensorList& inputs, size_t axis = 0,
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<ConcatenateOpDef>(make_ptr<ConcatenateOpDef>(
      ConcatenateOpDef::constrcutor_access_key(), inputs, axis, op_meta)) {}
};

class ConcatenateGradientOpDef : public OperatorDef {
 private:
  friend class ConcatenateGradientOp;
  struct constrcutor_access_key {};

 public:
  ConcatenateGradientOpDef(const constrcutor_access_key&, Tensor input, Tensor output,
                           Tensor grad_output, size_t axis, size_t offset,
                           const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ConcatenateGradientOp), {input, output, grad_output}, op_meta),
    _axis(axis), _offset(offset) {
    DoInferMeta();
    DoDeduceStates();
  }

  size_t get_axis() const {
    return _axis;
  }

  size_t get_offset() const {
    return _offset;
  }

  void set_offset(size_t offset) {
    _offset = offset;
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;  

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _axis;

  size_t _offset;
};

class ConcatenateGradientOp final : public OpWrapper<ConcatenateGradientOpDef> {
 public:
  ConcatenateGradientOp() : OpWrapper<ConcatenateGradientOpDef>() {}
  ConcatenateGradientOp(Tensor input, Tensor output, Tensor grad_output, size_t axis, size_t offset,
                        const OpMeta& op_meta = OpMeta())
  : OpWrapper<ConcatenateGradientOpDef>(make_ptr<ConcatenateGradientOpDef>(
      ConcatenateGradientOpDef::constrcutor_access_key(), input, output, grad_output,
      axis, offset, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

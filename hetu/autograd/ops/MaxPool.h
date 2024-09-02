#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class MaxPoolOpDef;
class MaxPoolOp;
class MaxPoolGradientOpDef;
class MaxPoolGradientOp;

class MaxPoolOpDef : public OperatorDef {
 private:
  friend class MaxPoolOp;
  struct constrcutor_access_key {};

 public:
  MaxPoolOpDef(const constrcutor_access_key&, Tensor input, size_t kernel_H,
               size_t kernel_W, size_t padding, size_t stride,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MaxPoolOp), {input}, op_meta),
    _kernel_H(kernel_H),
    _kernel_W(kernel_W),
    _padding(padding),
    _stride(stride) {
    DoInferMeta();
    DoDeduceStates();
  }

  size_t get_kernel_H() const {
    return _kernel_H;
  }

  size_t get_kernel_W() const {
    return _kernel_W;
  }

  size_t get_padding() const {
    return _padding;
  }

  size_t get_stride() const {
    return _stride;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _kernel_H;

  size_t _kernel_W;

  size_t _padding;

  size_t _stride;
};

class MaxPoolOp final : public OpWrapper<MaxPoolOpDef> {
 public:
  MaxPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, size_t padding,
            size_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MaxPoolOpDef>(
      make_ptr<MaxPoolOpDef>(MaxPoolOpDef::constrcutor_access_key(), input,
                             kernel_H, kernel_W, padding, stride, op_meta)) {}
};

class MaxPoolGradientOpDef : public OperatorDef {
 private:
  friend class MaxPoolGradientOp;
  struct constrcutor_access_key {};

 public:
  MaxPoolGradientOpDef(const constrcutor_access_key&, Tensor output,
                       Tensor output_grad, Tensor input, size_t kernel_H,
                       size_t kernel_W, size_t padding, size_t stride,
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(MaxPoolGradientOp), {output, output_grad, input},
                op_meta),
    _kernel_H(kernel_H),
    _kernel_W(kernel_W),
    _padding(padding),
    _stride(stride) {
    DoInferMeta();
    DoDeduceStates();
  }

  size_t get_kernel_H() const {
    return _kernel_H;
  }

  size_t get_kernel_W() const {
    return _kernel_W;
  }

  size_t get_padding() const {
    return _padding;
  }

  size_t get_stride() const {
    return _stride;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  size_t _kernel_H;

  size_t _kernel_W;

  size_t _padding;

  size_t _stride;
};

class MaxPoolGradientOp final : public OpWrapper<MaxPoolGradientOpDef> {
 public:
  MaxPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                    size_t kernel_H, size_t kernel_W, size_t padding,
                    size_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<MaxPoolGradientOpDef>(make_ptr<MaxPoolGradientOpDef>(
      MaxPoolGradientOpDef::constrcutor_access_key(), output, output_grad,
      input, kernel_H, kernel_W, padding, stride, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

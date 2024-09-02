#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class AvgPoolOpDef;
class AvgPoolOp;
class AvgPoolGradientOpDef;
class AvgPoolGradientOp;

class AvgPoolOpDef : public OperatorDef {
 private:
  friend class AvgPoolOp;
  struct constrcutor_access_key {};

 public:
  AvgPoolOpDef(const constrcutor_access_key&, Tensor input, size_t kernel_H,
               size_t kernel_W, size_t padding, size_t stride,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AvgPoolOp), {input}, op_meta),
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

class AvgPoolOp final : public OpWrapper<AvgPoolOpDef> {
 public:
  AvgPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, size_t padding,
            size_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AvgPoolOpDef>(
      make_ptr<AvgPoolOpDef>(AvgPoolOpDef::constrcutor_access_key(), input,
                             kernel_H, kernel_W, padding, stride, op_meta)) {}
};

class AvgPoolGradientOpDef : public OperatorDef {
 private:
  friend class AvgPoolGradientOp;
  struct constrcutor_access_key {};

 public:
  AvgPoolGradientOpDef(const constrcutor_access_key&, Tensor output,
                       Tensor output_grad, Tensor input, size_t kernel_H,
                       size_t kernel_W, size_t padding, size_t stride,
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AvgPoolGradientOp), {output, output_grad, input},
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

class AvgPoolGradientOp final : public OpWrapper<AvgPoolGradientOpDef> {
 public:
  AvgPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                    size_t kernel_H, size_t kernel_W, size_t padding,
                    size_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AvgPoolGradientOpDef>(make_ptr<AvgPoolGradientOpDef>(
      AvgPoolGradientOpDef::constrcutor_access_key(), output, output_grad,
      input, kernel_H, kernel_W, padding, stride, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

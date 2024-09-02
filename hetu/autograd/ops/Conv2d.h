#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class Conv2dOpDef;
class Conv2dOp;
class Conv2dGradientofFilterOpDef;
class Conv2dGradientofFilterOp;
class Conv2dGradientofDataOpDef;
class Conv2dGradientofDataOp;
class Conv2dAddBiasOpDef;
class Conv2dAddBiasOp;

class Conv2dOpDef : public OperatorDef {
 private:
  friend class Conv2dOp;
  struct constrcutor_access_key {};

 public:
  Conv2dOpDef(const constrcutor_access_key&, Tensor input, Tensor filter,
              int64_t padding, int64_t stride, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Conv2dOp), {input, filter}, op_meta) {
    _padding = {padding, padding};
    _stride = {stride, stride};
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _padding;

  HTShape _stride;
};

class Conv2dOp final : public OpWrapper<Conv2dOpDef> {
 public:
  Conv2dOp(Tensor input, Tensor filter, int64_t padding, int64_t stride,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<Conv2dOpDef>(
      make_ptr<Conv2dOpDef>(Conv2dOpDef::constrcutor_access_key(), input,
                            filter, padding, stride, op_meta)) {}
};

/*——————————————————————Conv2dGradientofFilter————————————————————————*/

class Conv2dGradientofFilterOpDef : public OperatorDef {
 private:
  friend class Conv2dGradientofFilterOp;
  struct constrcutor_access_key {};

 public:
  Conv2dGradientofFilterOpDef(const constrcutor_access_key&, Tensor input,
                              Tensor grad_output, Tensor filter,
                              const HTShape& padding, const HTStride& stride,
                              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Conv2dGradientofFilterOp), {input, grad_output, filter},
                op_meta),
    _padding(padding),
    _stride(stride) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;
  
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _padding;

  HTShape _stride;
};

class Conv2dGradientofFilterOp final
: public OpWrapper<Conv2dGradientofFilterOpDef> {
 public:
  Conv2dGradientofFilterOp(Tensor input, Tensor grad_output, Tensor filter,
                           const HTShape& padding, const HTStride& stride,
                           const OpMeta& op_meta = OpMeta())
  : OpWrapper<Conv2dGradientofFilterOpDef>(
      make_ptr<Conv2dGradientofFilterOpDef>(
        Conv2dGradientofFilterOpDef::constrcutor_access_key(), input,
        grad_output, filter, padding, stride, op_meta)) {}
};

/*——————————————————————Conv2dGradientofData————————————————————————*/

class Conv2dGradientofDataOpDef : public OperatorDef {
 private:
  friend class Conv2dGradientofDataOp;
  struct constrcutor_access_key {};

 public:
  Conv2dGradientofDataOpDef(const constrcutor_access_key&, Tensor filter,
                            Tensor grad_output, Tensor input,
                            const HTShape& padding, const HTStride& stride,
                            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Conv2dGradientofDataOp), {filter, grad_output, input},
                op_meta),
    _padding(padding),
    _stride(stride) {
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

 protected:
  void DoInferMeta() override;
  
  void DoDeduceStates() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _padding;

  HTShape _stride;
};

class Conv2dGradientofDataOp final
: public OpWrapper<Conv2dGradientofDataOpDef> {
 public:
  Conv2dGradientofDataOp(Tensor filter, Tensor grad_output, Tensor input,
                         const HTShape& padding, const HTStride& stride,
                         const OpMeta& op_meta = OpMeta())
  : OpWrapper<Conv2dGradientofDataOpDef>(make_ptr<Conv2dGradientofDataOpDef>(
      Conv2dGradientofDataOpDef::constrcutor_access_key(), filter, grad_output,
      input, padding, stride, op_meta)) {}
};

/*——————————————————————Conv2dAddBias————————————————————————*/

class Conv2dAddBiasOpDef : public OperatorDef {
 private:
  friend class Conv2dAddBiasOp;
  struct constrcutor_access_key {};

 public:
  Conv2dAddBiasOpDef(const constrcutor_access_key&, Tensor input, Tensor filter,
                     Tensor bias, int64_t padding, int64_t stride,
                     const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(Conv2dAddBiasOp), {input, filter, bias}, op_meta) {
    _padding = {padding, padding};
    _stride = {stride, stride};
    DoInferMeta();
    DoDeduceStates();
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

 protected:
  void DoInferMeta() override;

  void DoDeduceStates() override;
  
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _padding;

  HTShape _stride;
};

class Conv2dAddBiasOp final : public OpWrapper<Conv2dAddBiasOpDef> {
 public:
  Conv2dAddBiasOp(Tensor input, Tensor filter, Tensor bias, int64_t padding,
                  int64_t stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<Conv2dAddBiasOpDef>(make_ptr<Conv2dAddBiasOpDef>(
      Conv2dAddBiasOpDef::constrcutor_access_key(), input, filter, bias,
      padding, stride, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class AsStridedOpDef;
class AsStridedOp;
class AsStridedGradientOpDef;
class AsStridedGradientOp;

class AsStridedOpDef : public OperatorDef {
 private:
  friend class AsStridedOp;
  struct constrcutor_access_key {};

 public:
  AsStridedOpDef(const constrcutor_access_key&, Tensor input, 
                 HTShape outshape, HTShape stride,
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AsStridedOp), {input}, op_meta),
  _outshape(outshape),
  _stride(stride){
    DoInferMeta();
  }

  inline HTShape outshape() const{
    return _outshape;
  }

  inline HTShape get_stride() const{
    return _stride;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _outshape;

  HTShape _stride;
};

class AsStridedOp final : public OpWrapper<AsStridedOpDef> {
 public:
  AsStridedOp(Tensor input, HTShape outshape, HTShape stride, const OpMeta& op_meta = OpMeta())
  : OpWrapper<AsStridedOpDef>(make_ptr<AsStridedOpDef>(
      AsStridedOpDef::constrcutor_access_key(), input,outshape, stride, op_meta)) {}
};

class AsStridedGradientOpDef : public OperatorDef {
 private:
  friend class AsStridedGradientOp;
  struct constrcutor_access_key {};

 public:
  AsStridedGradientOpDef(const constrcutor_access_key&,
                         Tensor grad_output, Tensor input,
                         HTShape stride,
                         const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(AsStridedGradientOp), {grad_output, input},
                op_meta),
  _stride(stride) {
    DoInferMeta();
  }

  inline HTShape get_stride() const{
    return _stride;
  }


 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _stride;

};

class AsStridedGradientOp final
: public OpWrapper<AsStridedGradientOpDef> {
 public:
  AsStridedGradientOp(Tensor grad_output, Tensor input, HTShape stride,
                      const OpMeta& op_meta = OpMeta())
  : OpWrapper<AsStridedGradientOpDef>(
      make_ptr<AsStridedGradientOpDef>(
        AsStridedGradientOpDef::constrcutor_access_key(), grad_output, 
        input, stride, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

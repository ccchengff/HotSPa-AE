#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class InterpolateOpDef;
class InterpolateOp;
class InterpolateGradientOpDef;
class InterpolateGradientOp;

class InterpolateOpDef : public OperatorDef {
 private:
  friend class InterpolateOp;
  struct constrcutor_access_key {};

 public:
  InterpolateOpDef(const constrcutor_access_key&, Tensor input, const HTShape& outshape,
                   bool align_corners = false, double scale_factor = 0,   
                   const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(InterpolateOp), {input}, op_meta),
  _out_shape(outshape), 
  _align_corners(align_corners),
  _scale_factor(scale_factor) {
    DoInferMeta();
  }

  inline HTShape out_shape() const{
    return _out_shape;
  }

  inline bool align_corners() const{
    return _align_corners;
  }

  inline double scale_factor() const{
    return _scale_factor;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _out_shape;

  bool _align_corners;

  double _scale_factor;
};

class InterpolateOp final : public OpWrapper<InterpolateOpDef> {
 public:
  InterpolateOp(Tensor input, const HTShape& outshape,
                bool align_corners = false, double scale_factor = 0,
                const OpMeta& op_meta = OpMeta())
  : OpWrapper<InterpolateOpDef>(make_ptr<InterpolateOpDef>(
      InterpolateOpDef::constrcutor_access_key(), input, outshape, align_corners, scale_factor, op_meta)) {}
};

class InterpolateGradientOpDef : public OperatorDef {
 private:
  friend class InterpolateGradientOp;
  struct constrcutor_access_key {};

 public:
  InterpolateGradientOpDef(const constrcutor_access_key&,
                      Tensor grad_output, Tensor input,
                      bool align_corners = false, double scale_factor = 0,
                      const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(InterpolateGradientOp), {grad_output, input}, op_meta),
  _align_corners(align_corners),
  _scale_factor(scale_factor) {
    DoInferMeta();
  }

  inline bool align_corners() const{
    return _align_corners;
  }

  inline double scale_factor() const{
    return _scale_factor;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _align_corners;

  double _scale_factor;
};

class InterpolateGradientOp final
: public OpWrapper<InterpolateGradientOpDef> {
 public:
  InterpolateGradientOp(Tensor grad_output, Tensor input,
                        bool align_corners = false, double scale_factor = 0,
                        const OpMeta& op_meta = OpMeta())
  : OpWrapper<InterpolateGradientOpDef>(
      make_ptr<InterpolateGradientOpDef>(
        InterpolateGradientOpDef::constrcutor_access_key(), grad_output, input, align_corners, scale_factor, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

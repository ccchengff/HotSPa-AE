#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class NormOpDef;
class NormOp;
class NormGradientOpDef;
class NormGradientOp;

class NormOpDef : public OperatorDef {
 private:
  friend class NormOp;
  struct constrcutor_access_key {};

 public:
  NormOpDef(const constrcutor_access_key&, Tensor input, 
            int64_t p = 1, int64_t dim = 0, bool keepdim = false,
            const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NormOp), {input}, op_meta),
  _p(p),
  _dim(dim),
  _keepdim(keepdim) {
    DoInferMeta();
  }

  inline int64_t getp() const{
    return _p;
  }

  inline int64_t dim() const{
    return _dim;
  }

  inline bool keepdim() const{
    return _keepdim;
  }
  
 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _p;

  int64_t _dim;

  bool _keepdim;
};

class NormOp final : public OpWrapper<NormOpDef> {
 public:
  NormOp(Tensor input, int64_t p = 1, int64_t dim = 0, 
         bool keepdim =  false, const OpMeta& op_meta = OpMeta())
  : OpWrapper<NormOpDef>(make_ptr<NormOpDef>(
      NormOpDef::constrcutor_access_key(), input, p, dim, keepdim, op_meta)) {}
};

class NormGradientOpDef : public OperatorDef {
 private:
  friend class NormGradientOp;
  struct constrcutor_access_key {};

 public:
  NormGradientOpDef(const constrcutor_access_key&, Tensor input, Tensor output, 
                    Tensor grad_output, int64_t p, int64_t dim, 
                    const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(NormGradientOp), {input, output, grad_output}, op_meta),
  _p(p),
  _dim(dim) {
    DoInferMeta();
  }

  inline int64_t getp() const{
    return _p;
  }

  inline int64_t dim() const{
    return _dim;
  }

 protected:
  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  int64_t _p;

  int64_t _dim;
};

class NormGradientOp final : public OpWrapper<NormGradientOpDef> {
 public:
  NormGradientOp(Tensor input, Tensor output, Tensor grad_output, int64_t p, 
                 int64_t dim, const OpMeta& op_meta = OpMeta())
  : OpWrapper<NormGradientOpDef>(
      make_ptr<NormGradientOpDef>(NormGradientOpDef::constrcutor_access_key(),
                                  input, output, grad_output, p, dim, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

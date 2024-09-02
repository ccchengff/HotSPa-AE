#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class DataH2DOpDef;
class DataH2DOp;
class DataD2HOpDef;
class DataD2HOp;

class DataH2DOpDef : public OperatorDef {
 private:
  friend class DataH2DOp;
  struct constrcutor_access_key {};

 public:
  DataH2DOpDef(const constrcutor_access_key&, Tensor input,
               const Device& device, const OpMeta& op_meta = OpMeta());

 protected:
  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class DataH2DOp final : public OpWrapper<DataH2DOpDef> {
 public:
  DataH2DOp(Tensor input, const Device& device,
            const OpMeta& op_meta = OpMeta())
  : OpWrapper<DataH2DOpDef>(make_ptr<DataH2DOpDef>(
      DataH2DOpDef::constrcutor_access_key(), input, device, op_meta)) {}
};

class DataD2HOpDef : public OperatorDef {
 private:
  friend class DataD2HOp;
  struct constrcutor_access_key {};

 public:
  DataD2HOpDef(const constrcutor_access_key&, Tensor input,
               const OpMeta& op_meta = OpMeta());

 protected:
  bool DoPlaceToLocalDevice(const Device& placement,
                            StreamIndex stream_id) override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class DataD2HOp final : public OpWrapper<DataD2HOpDef> {
 public:
  DataD2HOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<DataD2HOpDef>(make_ptr<DataD2HOpDef>(
      DataD2HOpDef::constrcutor_access_key(), input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

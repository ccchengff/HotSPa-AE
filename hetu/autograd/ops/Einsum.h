#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

using OpDim = std::vector<std::string>;
using OpDimList = std::vector<OpDim>;
using LabelMap = std::unordered_map<std::string, int>;

class EinsumOpDef;
class EinsumOp;
class EinsumGradientOpDef;
class EinsumGradientOp;

class EinsumOpDef : public OperatorDef {
 private:
  friend class EinsumOp;
  struct constrcutor_access_key {};

 public:
  EinsumOpDef(const constrcutor_access_key&, const std::string& msg,
              const TensorList& inputs, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(EinsumOp), inputs, op_meta),
    _msg(msg),
    input_dims(),
    output_dims() {
    // ParseMsg();
    // HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    // AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()));
    DoInferMeta();
  }

  inline std::string fetch_msg() const {
    return _msg;
  }

  inline HTShape get_grad_shape() const {
    return _grad_shape;
  }

  void set_grad_shape(HTShape shape) {
    _grad_shape = shape;
  }

 protected:
  void ParseMsg();

  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _msg;

  std::vector<std::string> _input_msgs;

  std::string _output_msg;

  OpDimList input_dims;

  OpDimList output_dims;

  LabelMap num_labels;

  LabelMap output_labels_idx;

  int output_size;

  int num_output_labels;

  int elli_len;

  std::vector<int> input_elli_len;

  int elli_pos;

  HTShape _grad_shape;
};

class EinsumOp final : public OpWrapper<EinsumOpDef> {
 public:
  EinsumOp(const std::string& msg, const TensorList& inputs,
           const OpMeta& op_meta = OpMeta())
  : OpWrapper<EinsumOpDef>(make_ptr<EinsumOpDef>(
      EinsumOpDef::constrcutor_access_key(), msg, inputs, op_meta)) {}
};

class EinsumGradientOpDef : public OperatorDef {
 private:
  friend class EinsumGradientOp;
  struct constrcutor_access_key {};

 public:
  EinsumGradientOpDef(const constrcutor_access_key&, const std::string& msg,
                      const TensorList& inputs, Tensor ori_output,
                      Tensor ori_input, const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(EinsumGradientOp), inputs, op_meta),
    pred(ori_output),
    pred_in(ori_input),
    _msg(msg),
    input_dims(),
    output_dims() {
    // HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    // AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()));
    DoInferMeta();
  }

  inline std::string fetch_msg() const {
    return _msg;
  }

  Tensor pred;

  Tensor pred_in;

 protected:
  void ParseMsg(const HTShapeList& input_shapes);

  void DoInferMeta() override;

  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  std::string _msg;

  std::vector<std::string> _input_msgs;

  std::string _output_msg;

  OpDimList input_dims;

  OpDimList output_dims;

  LabelMap undefined_labels;

  LabelMap num_labels;

  LabelMap output_labels_idx;

  int output_size;

  int ori_output_size;

  int num_output_labels;

  int elli_len;

  std::vector<int> input_elli_len;

  int elli_pos;
};

class EinsumGradientOp final : public OpWrapper<EinsumGradientOpDef> {
 public:
  EinsumGradientOp(const std::string& msg, const TensorList& inputs,
                   Tensor ori_output, Tensor ori_input,
                   const OpMeta& op_meta = OpMeta())
  : OpWrapper<EinsumGradientOpDef>(make_ptr<EinsumGradientOpDef>(
      EinsumGradientOpDef::constrcutor_access_key(), msg, inputs, ori_output,
      ori_input, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

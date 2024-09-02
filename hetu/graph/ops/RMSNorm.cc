#include "hetu/graph/ops/RMSNorm.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void RMSNormOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  const NDArray x0 = input_indexs(0) >= 0 ? inputs.at(input_indexs(0)) : NDArray();
  const NDArray residual_ = input_indexs(1) >= 0 ? inputs.at(input_indexs(1)) : NDArray();
  const NDArray gamma = input_indexs(2) >= 0 ? inputs.at(input_indexs(2)) : NDArray();
  const NDArray beta_ = input_indexs(3) >= 0 ? inputs.at(input_indexs(3)) : NDArray();
  const NDArray rowscale_ = input_indexs(4) >= 0 ? inputs.at(input_indexs(4)) : NDArray();
  const NDArray colscale_ = input_indexs(5) >= 0 ? inputs.at(input_indexs(5)) : NDArray();
  const NDArray x0_subset_ = input_indexs(6) >= 0 ? inputs.at(input_indexs(6)) : NDArray();
  const NDArray z_subset_ = input_indexs(7) >= 0 ? inputs.at(input_indexs(7)) : NDArray();
  auto itype = x0->dtype();
  auto rtype = residual_.is_defined()
      ? residual_->dtype()
      : (residual_in_fp32() ? kFloat32 : x0->dtype());
  bool save_x = residual_.is_defined() || (dropout_p() > 0.f) || rowscale_.is_defined() || 
                colscale_.is_defined() || x0_subset_.is_defined() || (itype != rtype);
  NDArray z = output_indexs(0) >= 0 ? outputs.at(output_indexs(0)) : NDArray();
  NDArray x = (output_indexs(1) >= 0 && save_x) ? outputs.at(output_indexs(1)) : NDArray();
  NDArray dmask = output_indexs(2) >= 0 ? outputs.at(output_indexs(2)) : NDArray();
  NDArray mu = output_indexs(3) >= 0 ? outputs.at(output_indexs(3)) : NDArray();
  NDArray rsigma = output_indexs(4) >= 0 ? outputs.at(output_indexs(4)) : NDArray();

  int64_t hidden_size = gamma->numel();
  NDArray x0mat = x0.is_defined() ? NDArray::view(x0, {-1, hidden_size}) : x0;
  if (output_indexs(1) >= 0 && !save_x)
    outputs[output_indexs(1)] = x0;
  NDArray residualmat = residual_.is_defined() ? NDArray::view(residual_, {-1, hidden_size}) : residual_;
  NDArray rowscalemat = rowscale_.is_defined() ? NDArray::view(rowscale_, {-1}) : rowscale_;
  NDArray x0_subsetmat = x0_subset_.is_defined() ? NDArray::view(x0_subset_, {-1}) : x0_subset_;
  NDArray out_subsetmet = z_subset_.is_defined() ? NDArray::view(z_subset_, {-1}) : z_subset_;
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutAddLnFwd,
                               x0mat, residualmat, gamma, beta_, rowscalemat, colscale_, x0_subsetmat, out_subsetmet,
                               z, x, dmask, mu, rsigma, dropout_p(), epsilon(), rowscale_const(), z_numrows(),
                               residual_in_fp32(), is_rms_norm(), op->instantiation_ctx().stream());
}

TensorList RMSNormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  TensorList grad_tensors(op->num_inputs(), Tensor());
  // output_1, if input_5 ? input_0 : None, output_2, input_2, output_3, output_4, input_4, input_5 
  int64_t x0_numrows = 1;
  for (int i = 0; i < op->input(input_indexs(0))->ndim() - 1; ++i) {
    x0_numrows *= op->input(input_indexs(0))->shape(i);
  } 
  TensorList grads = MakeRMSNormGradientOp(grad_outputs.at(0),
                                           grad_outputs.at(1),
                                           output_indexs(1) >= 0 ? op->output(output_indexs(1)) : Tensor(),
                                           input_indexs(5) >= 0 ? op->input(input_indexs(0)) : Tensor(),
                                           output_indexs(2) >= 0 ? op->output(output_indexs(2)) : Tensor(),
                                           output_indexs(3) >= 0 ? op->output(output_indexs(3)) : Tensor(),
                                           output_indexs(4) >= 0 ? op->output(output_indexs(4)) : Tensor(),
                                           input_indexs(2) >= 0 ? op->input(input_indexs(2)) : Tensor(),
                                           input_indexs(4) >= 0 ? op->input(input_indexs(4)) : Tensor(),
                                           input_indexs(5) >= 0 ? op->input(input_indexs(5)) : Tensor(),
                                           input_indexs(6) >= 0 ? op->input(input_indexs(6)) : Tensor(),
                                           input_indexs(7) >= 0 ? op->input(input_indexs(7)) : Tensor(),
                                           dropout_p(), rowscale_const(), input_indexs(6) >= 0 ? x0_numrows : 0,
                                           input_indexs(1) >= 0 ? true : false, is_rms_norm(),
                                           op->grad_op_meta().set_name(op->grad_name()));
  if (input_indexs(0) >= 0 && op->input(input_indexs(0))->requires_grad())
    grad_tensors[input_indexs(0)] = grads[0];
  if (input_indexs(1) >= 0 && op->input(input_indexs(1))->requires_grad())
    grad_tensors[input_indexs(1)] = grads[1];
  if (input_indexs(2) >= 0 && op->input(input_indexs(2))->requires_grad())
    grad_tensors[input_indexs(2)] = grads[2];
  if (input_indexs(3) >= 0 && op->input(input_indexs(3))->requires_grad())
    grad_tensors[input_indexs(3)] = grads[3];
  if (input_indexs(5) >= 0 && op->input(input_indexs(5))->requires_grad())
    grad_tensors[input_indexs(5)] = grads[5];
  return grad_tensors;
}

HTShapeList RMSNormOpImpl::DoInferShape(Operator& op, 
                                        const HTShapeList& input_shapes, 
                                        RuntimeContext& ctx) const {
  HTShapeList out_shapes = {};
  int ptr = 0;
  HTShape x0 = input_indexs(0) >= 0 ? input_shapes.at(input_indexs(0)) : HTShape();
  HTShape residual_ = input_indexs(1) >= 0 ? input_shapes.at(input_indexs(1)) : HTShape();
  HTShape gamma = input_indexs(2) >= 0 ? input_shapes.at(input_indexs(2)) : HTShape();
  HTShape beta_ = input_indexs(3) >= 0 ? input_shapes.at(input_indexs(3)) : HTShape();
  HTShape rowscale_ = input_indexs(4) >= 0 ? input_shapes.at(input_indexs(4)) : HTShape();
  HTShape colscale_ = input_indexs(5) >= 0 ? input_shapes.at(input_indexs(5)) : HTShape();
  HTShape x0_subset_ = input_indexs(6) >= 0 ? input_shapes.at(input_indexs(6)) : HTShape();
  HTShape z_subset_ = input_indexs(7) >= 0 ? input_shapes.at(input_indexs(7)) : HTShape();
  int64_t hidden_size = NumEl(gamma);
  HTShape x0mat_shape = {NumEl(x0) / hidden_size, hidden_size}, x0_subset_shape = {};
  if (x0_subset_.size() > 0)
    x0_subset_shape = {NumEl(x0_subset_) / hidden_size, hidden_size};
  HTShape sizes {!(x0_subset_.size() > 0) ? x0mat_shape[0] : x0_subset_shape[0], x0mat_shape[1]};
  const int rows = sizes[0];
  const int cols = sizes[1];
  HTShape zmat_shape = z_subset_.size() > 0 ? HTShape{z_numrows(), cols} : sizes;
  int64_t numel = 1;
  for (int i = 1; i < x0.size(); ++i)
    numel *= x0[i];
  HTShape x_shape = {NumEl(sizes) / numel};
  HTShape z_shape = {NumEl(zmat_shape) / numel};
  for (int i = 1; i < x0.size(); ++i) {
    x_shape.emplace_back(x0[i]);
    z_shape.emplace_back(x0[i]);
  }
  out_shapes.emplace_back(z_shape);
  if (output_indexs(1) >= 0)
    out_shapes.emplace_back(x_shape);
  if (output_indexs(2) >= 0)
    out_shapes.emplace_back(x_shape);
  out_shapes.emplace_back(HTShape{ rows });
  out_shapes.emplace_back(HTShape{ rows });
  return out_shapes;
}

void RMSNormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                                   const OpMeta& op_meta) const {
  for (auto output : outputs)
    output->set_distributed_states(inputs.at(0)->get_distributed_states()); 
}

void RMSNormOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                      TensorList& outputs, const OpMeta& op_meta) const {
  for (auto output : outputs)
    output->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void RMSNormGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                      NDArrayList& outputs, RuntimeContext& ctx) const {
  for (auto input : op->inputs()) {
    HT_LOG_TRACE << "rms gradient input " << input << ", ds = " << input->get_distributed_states().ds_info() << ", shape = " << input->shape()
      << ", producer = " << input->producer(); 
  }
  const NDArray dz = input_indexs(0) >= 0 ? inputs.at(input_indexs(0)) : NDArray();
  const NDArray dx_ = input_indexs(1) >= 0 ? inputs.at(input_indexs(1)) : NDArray();
  const NDArray x = input_indexs(2) >= 0 ? inputs.at(input_indexs(2)) : NDArray();
  const NDArray x0_ = input_indexs(3) >= 0 ? inputs.at(input_indexs(3)) : NDArray();
  const NDArray dmask_ = input_indexs(4) >= 0 ? inputs.at(input_indexs(4)) : NDArray();
  const NDArray mu = input_indexs(5) >= 0 ? inputs.at(input_indexs(5)) : NDArray();
  const NDArray rsigma = input_indexs(6) >= 0 ? inputs.at(input_indexs(6)) : NDArray();
  const NDArray gamma = input_indexs(7) >= 0 ? inputs.at(input_indexs(7)) : NDArray();
  const NDArray rowscale_ = input_indexs(8) >= 0 ? inputs.at(input_indexs(8)) : NDArray();
  const NDArray colscale_ = input_indexs(9) >= 0 ? inputs.at(input_indexs(9)) : NDArray();
  const NDArray x0_subset_ = input_indexs(10) >= 0 ? inputs.at(input_indexs(10)) : NDArray();
  const NDArray z_subset_ = input_indexs(11) >= 0 ? inputs.at(input_indexs(11)) : NDArray();
  NDArray dx0 = output_indexs(0) >= 0 ? outputs.at(output_indexs(0)) : NDArray();
  NDArray dresidual = output_indexs(1) >= 0 ? outputs.at(output_indexs(1)) : NDArray();
  NDArray dgamma = output_indexs(2) >= 0 ? outputs.at(output_indexs(2)) : NDArray();
  NDArray dbeta = output_indexs(3) >= 0 ? outputs.at(output_indexs(3)) : NDArray();
  NDArray dcolscale = output_indexs(4) >= 0 ? outputs.at(output_indexs(4)) : NDArray();

  int64_t hidden_size = gamma->numel();
  NDArray xmat = x.is_defined() ? NDArray::view(x, {-1, hidden_size}) : x;
  NDArray dzmat = dz.is_defined() ? NDArray::view(dz, {-1, hidden_size}) : dz;
  NDArray dxmat = dx_.is_defined() ? NDArray::view(dx_, {-1, hidden_size}) : dx_;
  NDArray x0mat = x0_.is_defined() ? NDArray::view(x0_, {-1, hidden_size}) : x0_;
  NDArray rowscalemat = rowscale_.is_defined() ? NDArray::view(rowscale_, {-1}) : rowscale_;
  NDArray x0_subsetmat = x0_subset_.is_defined() ? NDArray::view(x0_subset_, {-1}) : x0_subset_;
  NDArray out_subsetmat = z_subset_.is_defined() ? NDArray::view(z_subset_, {-1}) : z_subset_;

  NDArray dgamma_part = NDArray();
  NDArray dbeta_part = NDArray(); 
  NDArray dcolscale_part = NDArray();

  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::DropoutAddLnBwd, dzmat, dxmat, xmat, x0mat, dmask_, mu,
                               rsigma, gamma, rowscalemat, colscale_, x0_subsetmat, out_subsetmat,
                               dx0, dresidual, dgamma, dbeta, dgamma_part, dbeta_part, dcolscale,
                               dcolscale_part, dropout_p(), rowscale_const(), x0_numrows(),
                               has_residual(), is_rms_norm(), op->instantiation_ctx().stream());
}

// workaround: need to care about all input cases
void RMSNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                                           const OpMeta& op_meta) const {
  const DistributedStates& ds_output_grad = inputs.at(0)->get_distributed_states();
  int reduce_dim = inputs.at(0)->ndim() - 1;
  HTAxes axes(reduce_dim);
  HTKeepDims keepdims(reduce_dim);
  for (int d = 0; d < reduce_dim; d++) {
    axes[d] = d;
    keepdims[d] = false;
  }
  DistributedStates ds_gamma_scale = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes, keepdims);
  //HT_LOG_TRACE << "ds_output_grad = " << ds_output_grad.ds_info() << ", ds_gamma_scale = " << ds_gamma_scale.ds_info() << ", output size = " << outputs.size() << ", indx2 = " << output_indexs(2);
  outputs.at(0)->set_distributed_states(ds_output_grad);
  outputs.at(output_indexs(2))->set_distributed_states(ds_gamma_scale);
  outputs.at(output_indexs(3))->set_distributed_states(ds_gamma_scale);
  //HT_LOG_TRACE << "RMSNormGradientOpImpl do gradient end";
}

void RMSNormGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                              TensorList& outputs, const OpMeta& op_meta) const {
  HT_ASSERT(inputs_hetero_dim.at(0) >= 0)
    << "Currently not support complex hetero dim deducing"
    << ", the hetero dim should be spilt and reduced to partial";
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
  outputs.at(1)->cur_ds_union().set_hetero_dim(-2);
  outputs.at(2)->cur_ds_union().set_hetero_dim(-2);
}

HTShapeList RMSNormGradientOpImpl::DoInferShape(Operator& op, 
                                                const HTShapeList& input_shapes, 
                                                RuntimeContext& ctx) const {
  HT_LOG_TRACE << "GradInferShape";
  HTShapeList out_shapes = {};
  HTShape dz = input_indexs(0) >= 0 ? input_shapes.at(input_indexs(0)) : HTShape();
  HTShape dx_ = input_indexs(1) >= 0 ? input_shapes.at(input_indexs(1)) : HTShape();
  HTShape x = input_indexs(2) >= 0 ? input_shapes.at(input_indexs(2)) : HTShape();
  HTShape x0_ = input_indexs(3) >= 0 ? input_shapes.at(input_indexs(3)) : HTShape();
  HTShape dmask_ = input_indexs(4) >= 0 ? input_shapes.at(input_indexs(4)) : HTShape();
  HTShape mu = input_indexs(5) >= 0 ? input_shapes.at(input_indexs(5)) : HTShape();
  HTShape rsigma = input_indexs(6) >= 0 ? input_shapes.at(input_indexs(6)) : HTShape();
  HTShape gamma = input_indexs(7) >= 0 ? input_shapes.at(input_indexs(7)) : HTShape();
  HTShape rowscale_ = input_indexs(8) >= 0 ? input_shapes.at(input_indexs(8)) : HTShape();
  HTShape colscale_ = input_indexs(9) >= 0 ? input_shapes.at(input_indexs(9)) : HTShape();
  HTShape x0_subset_ = input_indexs(10) >= 0 ? input_shapes.at(input_indexs(10)) : HTShape();
  HTShape z_subset_ = input_indexs(11) >= 0 ? input_shapes.at(input_indexs(11)) : HTShape();

  auto sizes = x;
  auto rows = sizes[0];
  auto cols = sizes[1];
  HTShape x0_sizes {!x0_subset_.size() > 0 ? rows : x0_numrows(), cols};
  out_shapes.emplace_back(x);
  if (has_residual()) {
    out_shapes.emplace_back(x);
  } 
  out_shapes.emplace_back(gamma);
  out_shapes.emplace_back(gamma);
  if (output_indexs(4) >= 0) {
    out_shapes.emplace_back(colscale_);
  }
  HT_LOG_TRACE << "GradInferShape-end";
  return out_shapes;
}

TensorList MakeRMSNormOp(Tensor x0, Tensor residual_, Tensor gamma,
                         Tensor beta_, Tensor rowscale_, Tensor colscale_,
                         Tensor x0_subset_, Tensor z_subset_, 
                         const float dropout_p, const float epsilon,
                         const float rowscale_const, const int64_t z_numrows,
                         bool residual_in_fp32, bool prenorm, 
                         bool is_rms_norm, bool return_dmask, 
                         OpMeta op_meta) {
  auto itype = x0->dtype();
  auto rtype = residual_.is_defined()
        ? residual_->dtype()
        : (residual_in_fp32 ? kFloat32 : x0->dtype());
  std::vector<int> output_indexs(5, -1);
  int ptr = 0;
  bool save_x = residual_.is_defined() || (dropout_p > 0.f) || rowscale_.is_defined() || 
                colscale_.is_defined() || x0_subset_.is_defined() || (itype != rtype);
  output_indexs[0] = ptr++;
  output_indexs[1] = ptr++;
  // if (save_x)
  //   output_indexs[1] = ptr++;
  if (dropout_p > 0.f)
    output_indexs[2] = ptr++;
  output_indexs[3] = ptr++;
  output_indexs[4] = ptr++;

  TensorList inputs = {std::move(x0), std::move(residual_), std::move(gamma),
                       std::move(beta_), std::move(rowscale_), std::move(colscale_),
                       std::move(x0_subset_), std::move(z_subset_)};
  TensorList inputs_ = {};
  std::vector<int> input_indexs(8, -1);
  ptr = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs[i].is_defined()) {
      input_indexs[i] = ptr++;
      inputs_.emplace_back(inputs[i]);
    }
  }
  return Graph::MakeOp(
        std::make_shared<RMSNormOpImpl>(dropout_p, epsilon, 
                                        rowscale_const, z_numrows,
                                        residual_in_fp32, prenorm, 
                                        is_rms_norm, return_dmask,
                                        input_indexs, output_indexs),
        std::move(inputs_),
        std::move(op_meta))->outputs();
}

TensorList MakeRMSNormGradientOp(Tensor dz, Tensor dx_, Tensor x, Tensor x0_,     
                                 Tensor dmask_, Tensor mu, Tensor rsigma, Tensor gamma,   
                                 Tensor rowscale_, Tensor colscale_, Tensor x0_subset_,  
                                 Tensor z_subset_, const float dropout_p, const float rowscale_const,
                                 const int64_t x0_numrows, const bool has_residual,
                                 bool is_rms_norm, OpMeta op_meta) {
  HT_LOG_TRACE << dz;
  HT_LOG_TRACE << dx_;
  HT_LOG_TRACE << x;
  HT_LOG_TRACE << x0_;
  HT_LOG_TRACE << dmask_;
  HT_LOG_TRACE << mu;
  HT_LOG_TRACE << rsigma;
  HT_LOG_TRACE << gamma;
  HT_LOG_TRACE << rowscale_;
  HT_LOG_TRACE << colscale_;
  HT_LOG_TRACE << x0_subset_;
  HT_LOG_TRACE << z_subset_;
  std::vector<int> output_indexs(5, -1);
  int ptr = 0;
  output_indexs[0] = ptr++;
  if (has_residual) {
    output_indexs[1] = ptr++;
  } 
  output_indexs[2] = ptr++;
  output_indexs[3] = ptr++;
  if (colscale_.is_defined()) {
    output_indexs[4] = ptr++;
  }
  HT_LOG_TRACE << "U1";
  TensorList inputs = {std::move(dz), std::move(dx_), std::move(x), std::move(x0_),
                       std::move(dmask_), std::move(mu), std::move(rsigma), std::move(gamma),
                       std::move(rowscale_), std::move(colscale_), std::move(x0_subset_), std::move(z_subset_)};
  TensorList inputs_ = {};
  HT_LOG_TRACE << "U2";
  std::vector<int> input_indexs(12, -1);
  ptr = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs[i].is_defined()) {
      input_indexs[i] = ptr++;
      inputs_.emplace_back(inputs[i]);
    }
  }
   HT_LOG_TRACE << "U3";
  return Graph::MakeOp(
         std::make_shared<RMSNormGradientOpImpl>(dropout_p, rowscale_const, x0_numrows, 
                                                 has_residual, is_rms_norm, input_indexs,
                                                 output_indexs),
         std::move(inputs_),
         std::move(op_meta))->outputs();
}


} // namespace graph
} // namespace hetu


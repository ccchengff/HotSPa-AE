#include "hetu/autograd/ops/MatMul2.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void MatMul2OpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HTShape out = outputs.at(0)->shape();
  out.pop_back(); out.pop_back();
  HTShape bshape_a = out;
  bshape_a.emplace_back(inputs.at(0)->shape(inputs.at(0)->ndim() - 2));
  bshape_a.emplace_back(inputs.at(0)->shape(inputs.at(0)->ndim() - 1));
  HTShape bshape_b = out;
  bshape_b.emplace_back(inputs.at(1)->shape(inputs.at(1)->ndim() - 2));
  bshape_b.emplace_back(inputs.at(1)->shape(inputs.at(1)->ndim() - 1));
  NDArray broadcast_a = NDArray::empty(bshape_a, inputs.at(0)->device(), inputs.at(0)->dtype());
  NDArray broadcast_b = NDArray::empty(bshape_b, inputs.at(1)->device(), inputs.at(1)->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    broadcast_a, HTAxes(), stream());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(1),
                                    broadcast_b, HTAxes(), stream());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::BatchMatMul,
                               broadcast_a, trans_a(), broadcast_b, trans_b(),
                               outputs.at(0), stream());
}

TensorList MatMul2OpDef::DoGradient(const TensorList& grad_outputs) {
  const Tensor& grad_c = grad_outputs.at(0);
  Tensor& a = _inputs[0];
  Tensor& b = _inputs[1];
  Tensor grad_a;
  Tensor grad_b;
  auto g_op_meta = grad_op_meta();
  if (!trans_a() && !trans_b()) {
    // case 1: c = MatMul(a, b)
    // grad_a = MatMul(grad_c, b^T), grad_b = MatMul(a^T, grad_c)
    grad_a = MatMul2GradientOp(grad_c, b, 0, a, _outputs[0], false, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMul2GradientOp(a, grad_c, 1, b, _outputs[0], true, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else if (trans_a() && !trans_b()) {
    // case 2: c = MatMul(a^T, b)
    // grad_a = MatMul(b, grad_c^T), grad_b = MatMul(a, grad_c)
    grad_a = MatMul2GradientOp(b, grad_c, 0, a, _outputs[0], false, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMul2GradientOp(a, grad_c, 1, b, _outputs[0], false, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else if (!trans_a() && trans_b()) {
    // case 3: c = MatMul(a, b^T)
    // grad_a = MatMul(grad_c, b), grad_b = MatMul(grad_c^T, a)
    grad_a = MatMul2GradientOp(grad_c, b, 0, a, _outputs[0], false, false, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMul2GradientOp(grad_c, a, 1, b, _outputs[0], true, false, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  } else {
    // case 4: c = MatMul(a^T, b^T)
    // grad_a = MatMul(b^T, grad_c^T), grad_b = MatMul(grad_c^T, a^T)
    grad_a = MatMul2GradientOp(b, grad_c, 0, a, _outputs[0], true, true, g_op_meta.set_name(grad_name(0)))
               ->output(0);
    grad_b = MatMul2GradientOp(grad_c, a, 1, b, _outputs[0], true, true, g_op_meta.set_name(grad_name(1)))
               ->output(0);
  }
  return {grad_a, grad_b};
}

void MatMul2OpDef::DoInferMeta() {
  auto a = _inputs[0];
  auto b = _inputs[1];
      if (a->has_shape() && b->has_shape()) {
      HTShape tmp_a = a->shape();
      HTShape tmp_b = b->shape();
      tmp_a.pop_back(); 
      tmp_a.pop_back();
      tmp_a.emplace_back(1);
      tmp_b.pop_back();
      tmp_b.pop_back();
      tmp_b.emplace_back(1);
      HTShape out = NDArrayMeta::Broadcast(tmp_a, tmp_b);
      out.pop_back();
      out.emplace_back(a->shape(trans_a() ? a->ndim() - 1 : a->ndim() - 2));
      out.emplace_back(b->shape(trans_b() ? a->ndim() - 2 : a->ndim() - 1));
      AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(out));
    }
    else
      AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape({}));
}

HTShapeList MatMul2OpDef::DoInferShape(const HTShapeList& input_shapes) {
  const HTShape &a = input_shapes.at(0);
  const HTShape &b = input_shapes.at(1);
  HT_ASSERT(a.size() >= 2 && b.size() >= 2 &&
            a.at(trans_a() ? a.size() - 2 : a.size() - 1) == b.at(trans_b() ? b.size() - 1 : b.size() - 2))
    << "Failed to infer shape for the \"" << type() << "\" operation "
    << "(with name \"" << name() << "\"): "
    << "Invalid input shapes: " << a << " (transpose_a = " << trans_a()
    << ") vs. " << b << " (transpose_b = " << trans_b() << "). ";
  HTShape tmp_a = a;
  HTShape tmp_b = b;
  tmp_a.pop_back(); 
  tmp_a.pop_back();
  tmp_a.emplace_back(1);
  tmp_b.pop_back();
  tmp_b.pop_back();
  tmp_b.emplace_back(1);
  HTShape out = NDArrayMeta::Broadcast(tmp_a, tmp_b);
  out.pop_back();
  HTShape bshape_a = out;
  HTShape bshape_b = out;
  out.emplace_back(a.at(trans_a() ? a.size() - 1 : a.size() - 2));
  out.emplace_back(b.at(trans_b() ? b.size() - 2 : b.size() - 1));
  bshape_a.emplace_back(a.at(a.size() - 2));
  bshape_a.emplace_back(a.at(a.size() - 1));
  bshape_b.emplace_back(b.at(b.size() - 2));
  bshape_b.emplace_back(b.at(b.size() - 1));
  for (size_t idx = 0; idx < 2; ++idx) {
    HTShape input_shape = input_shapes[idx];
    HTShape output_shape = idx ? bshape_b : bshape_a;
    size_t ndim = output_shape.size();
    HT_ASSERT(input_shape.size() <= ndim);
    size_t diff = ndim - input_shape.size();
    HTAxes add_axes(diff);
    HTKeepDims keep_dims(diff);
    size_t len = diff + input_shape.size();
    HTShape n_input_shape(len);
    for (size_t i = 0; i < diff; ++i) {
      add_axes[i] = i;
      keep_dims[i] = false;
      n_input_shape[i] = 1;
    }
    for (size_t i = diff; i < len; ++i) {
      n_input_shape[i] = input_shape[i - diff];
    }
    for (size_t i = 0; i < ndim; ++i) {
      if (output_shape[i] == -1) {
        output_shape[i] = n_input_shape[i];
      }
      HT_ASSERT(output_shape[i] > 0);
      HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
      if (i >= diff && input_shape[i] == 1 && output_shape[i] > 1) {
        add_axes.emplace_back(i);
        keep_dims.emplace_back(true);
      }
    }
    set_grad_axes(add_axes, idx);
    set_grad_keep_dims(keep_dims, idx);
  }
  return {out};
}

void MatMul2GradientOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                                    RuntimeContext& ctx) {
  HTShape a = inputs.at(0)->shape();
  HTShape b = inputs.at(1)->shape();
  HTShape out = a.size() > b.size() ? a: b;
  out.pop_back(); 
  out.pop_back();
  HTShape bshape_a = out;
  bshape_a.emplace_back(a.at(a.size() - 2));
  bshape_a.emplace_back(a.at(a.size() - 1));
  HTShape bshape_b = out;
  bshape_b.emplace_back(b.at(b.size() - 2));
  bshape_b.emplace_back(b.at(b.size() - 1));
  out.emplace_back(a.at(trans_a() ? a.size() - 1 : a.size() - 2));
  out.emplace_back(b.at(trans_b() ? b.size() - 2 : b.size() - 1));
  NDArray broadcast_a = (a == bshape_a)? inputs.at(0): NDArray::empty(bshape_a, inputs.at(0)->device(), inputs.at(0)->dtype());
  if (a != bshape_a) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(0),
                                    broadcast_a, HTAxes(), stream());
  }
  NDArray broadcast_b = (b == bshape_b)? inputs.at(1): NDArray::empty(bshape_b, inputs.at(1)->device(), inputs.at(1)->dtype());
  if (b != bshape_b) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(1),
                                    broadcast_b, HTAxes(), stream());
  }
  NDArray unreduced = axes().empty() ? outputs.at(0) : NDArray::empty(out, inputs.at(0)->device(), inputs.at(0)->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(), hetu::impl::BatchMatMul,
                               broadcast_a, trans_a(), broadcast_b, trans_b(),
                               unreduced, stream());
  if (!axes().empty()) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    placement().type(), type(), hetu::impl::ReduceSum, unreduced,
    outputs.at(0), axes().data(), axes().size(), stream());
  }
}

void MatMul2GradientOpDef::DoInferMeta() {
  AddOutput(_inputs[2]->meta());
}

HTShapeList MatMul2GradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  MatMul2Op& input_ptr =
    reinterpret_cast<MatMul2Op&>(_inputs[3]->producer());
  if (input_ptr) {
    set_axes(input_ptr->grad_axes(index()));
    set_keep_dims(input_ptr->grad_keep_dims(index()));
  }
  return {input_shapes.at(2)};
}

} // namespace autograd
} // namespace hetu

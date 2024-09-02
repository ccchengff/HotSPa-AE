#include "hetu/autograd/ops/BroadcastShape.h"
#include "hetu/autograd/ops/ReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void BroadcastShapeOpDef::DoCompute(const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::BroadcastShape, inputs.at(0),
                                  outputs.at(0), get_add_axes(), stream());
}

TensorList BroadcastShapeOpDef::DoGradient(const TensorList& grad_outputs) {
  auto grad_input = ReduceSumOp(grad_outputs.at(0), HTAxes(), HTKeepDims(),
                                grad_op_meta().set_name(grad_name()))
                      ->output(0);
  return {grad_input};
}

HTShapeList BroadcastShapeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  HTShape input_shape = input_shapes.at(0);
  HTShape output_shape = get_shape();
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim)
    << "input dim > output dim, input size is " << input_shape
    << " and output size is " << output_shape;
  size_t diff = ndim - input_shape.size();
  HTShape add_axes = get_add_axes();
  if (add_axes.size() > 0) {
    size_t asize = add_axes.size();
    HT_ASSERT(diff == asize ||
              (input_shape.size() == 1 && input_shape[0] == 1));
    HTKeepDims keep_dims(asize);
    for (size_t i = 0; i < asize; ++i) {
      keep_dims[i] = false;
      HT_ASSERT((size_t) add_axes[i] < ndim);
    }
    int64_t in_ind = 0;
    for (size_t i = 0; i < ndim; ++i) {
      bool flag = false;
      for (size_t j = 0; j < asize; ++j) {
        if (i == (size_t) add_axes[j]) {
          flag = true;
          break;
        }
      }
      if (!flag) {
        HT_ASSERT(input_shape[in_ind] == output_shape[i])
          << "input shape:" << input_shape << ",output shape:" << output_shape
          << ",add_axes:" << add_axes;
        in_ind++;
      }
    }
    set_grad_axes(add_axes);
    set_grad_keep_dims(keep_dims);
    // ReduceSumOp& input_ptr = reinterpret_cast<ReduceSumOp&>(grad_input);
    // if (input_ptr) {
    //   input_ptr->set_axes(add_axes);
    //   input_ptr->set_keepdims(keep_dims);
    // }
  } else {
    add_axes.resize(diff);
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
      HT_ASSERT(output_shape[i] > 0) << "has Invalid shape.";
      HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i])
        << "input shape can't broadcast to output shape.";
      if (i >= diff && n_input_shape[i] == 1 && output_shape[i] > 1) {
        add_axes.emplace_back(i);
        keep_dims.emplace_back(true);
      }
    }
    set_grad_axes(add_axes);
    set_grad_keep_dims(keep_dims);
    // ReduceSumOp& input_ptr = reinterpret_cast<ReduceSumOp&>(grad_input);
    // if (input_ptr) {
    //   input_ptr->set_axes(add_axes);
    //   input_ptr->set_keepdims(keep_dims);
    // }
  }
  return {output_shape};
}

} // namespace autograd
} // namespace hetu
#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

void NormCpu(const NDArray& input, NDArray& output, int64_t dim, int64_t p, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "NormCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
    [stream, input, output, dim, p]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
      dnnl::memory::dims in_shape = input->shape();
      dnnl::memory::dims in_stride = input->stride();
      dnnl::memory::dims out_shape = input->shape();
      dnnl::memory::dims out_stride(input->ndim());
      out_shape[dim] = 1;
      size_t stride_size = 1;
      for (int i = input->ndim() - 1; i >= 0; i--) {
        out_stride[i] = stride_size;
        stride_size *= out_shape[i];
      }
      auto src_md = dnnl::memory::desc(in_shape, dnnltype, in_stride);
      auto dst_md = dnnl::memory::desc(out_shape, dnnltype, out_stride);

      auto src_mem = dnnl::memory(src_md, eng, input->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

      if (in_shape == out_shape)
        hetu::cpu::read_from_dnnl_memory(output->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_norm_lp_sum, src_md, dst_md, float(p), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        dnnl::stream engine_stream(eng);
        reduction_prim.execute(engine_stream, reduction_args);
        engine_stream.wait();
    }
  },"Norm");
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
spec_t sgn(spec_t x) {
    if (x == 0.0)
        return 0.0;
    return x / std::abs(x);
}

template <typename spec_t>
void norm_gradient_cpu(const spec_t *input, const spec_t *norm,
                       const spec_t *grad, spec_t *output, int64_t p,
                       size_t reduce_dim_size,
                       size_t after_dim_size, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    int na = idx / (reduce_dim_size * after_dim_size);
    int nc = (idx % (reduce_dim_size * after_dim_size)) % after_dim_size;
    int idx_y = na * after_dim_size + nc;

    spec_t input_val = input[idx];
    spec_t grad_val = grad[idx_y];

    if (p == 1) {
        output[idx] = sgn(input_val) * grad_val;
    } else if (p == 2) {
        spec_t norm_val = norm[idx_y];
        if (norm_val == 0)
            output[idx] = 0;
        else
            output[idx] = grad_val * input_val / norm_val;
    } else if (p > 2) {
        spec_t norm_val = norm[idx_y];
        if (norm_val == 0)
            output[idx] = 0;
        else
            output[idx] = input_val * std::pow(std::abs(input_val), p - 2) * grad_val
                          / std::pow(norm_val, p - 1);
    }
  }
}

void NormGradientCpu(const NDArray& input, const NDArray& output, const NDArray& output_grad,
                     NDArray& input_grad, int64_t dim, int64_t p, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  CPUStream cpu_stream(stream);

  size_t reduce_dim_size, after_dim_size, size;
  reduce_dim_size = after_dim_size = size = 1;
  for (size_t i = 0; i < input->ndim(); ++i) {
      size *= input->shape(i);
      if (i == size_t(dim))
          reduce_dim_size = input->shape(i);
      else if (i > size_t(dim))
          after_dim_size *= input->shape(i);
  }

  if (size == 0)
    return;

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "NormGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, output_grad, input_grad, p, reduce_dim_size, after_dim_size, size]() {
      norm_gradient_cpu<spec_t>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), 
      input_grad->data_ptr<spec_t>(), p, reduce_dim_size, after_dim_size, size);
      },"NormGradient");
    });
  NDArray::MarkUsedBy({input, output, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void sigmoid_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = 1.0 / (1.0 + 1.0 / std::exp(input[idx]));
  }
}

template <typename spec_t>
void sigmoid_cpu(const spec_t* input, size_t size, spec_t* output,
                 int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = 1.0 / (1.0 + 1.0 / std::exp(input[i_idx]));
  }
}

template <typename spec_t>
void sigmoid_cpu(const spec_t* input, size_t size, spec_t* output,
                 int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                 const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = 1.0 / (1.0 + 1.0 / std::exp(input[i_idx]));
  }
}

void SigmoidCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SigmoidCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, 0);
          auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
          auto mat_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
          auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

          auto Sigmoid_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_logistic, mat_md, mat_md, float(0.0), float(0.0));
          auto Sigmoid = dnnl::eltwise_forward(Sigmoid_pd);

          std::unordered_map<int, dnnl::memory> sigmoid_args;
          sigmoid_args.insert({DNNL_ARG_SRC, src_mem});
          sigmoid_args.insert({DNNL_ARG_DST, dst_mem});
          dnnl::stream engine_stream(eng);
          Sigmoid.execute(engine_stream, sigmoid_args);
          engine_stream.wait();
      }, "Sigmoid");   
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
void sigmoid_grad_cpu(const spec_t* output_grad, const spec_t* output,
                      size_t size, spec_t* input_grad) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) 
    input_grad[idx] = output_grad[idx] * output[idx] * (1 - output[idx]);
}

template <typename spec_t>
void sigmoid_grad_cpu(const spec_t* output_grad, const spec_t* output,
                      size_t size, spec_t* input_grad,
                      int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                      const int64_t* stride_in, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t og_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    int64_t ig_idx = hetu::impl::get_index(idx, ndims, stride_in, c_shape);
    input_grad[ig_idx] = output_grad[og_idx] * output[o_idx] * (1 - output[o_idx]);
  }
}

void SigmoidGradientCpu(const NDArray& out_grad, const NDArray& output, NDArray& in_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(out_grad);
  HT_ASSERT_SAME_DEVICE(out_grad, output);
  HT_ASSERT_SAME_DEVICE(out_grad, in_grad);
  HT_ASSERT_EXCHANGABLE(out_grad, output);
  HT_ASSERT_EXCHANGABLE(out_grad, in_grad);

  size_t size = output->numel();
  if (size == 0)
    return;

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "SigmoidGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [output, out_grad, in_grad, size]() {
        if (out_grad->is_contiguous() && output->is_contiguous() && in_grad->is_contiguous()) {
          sigmoid_grad_cpu<spec_t>(
            out_grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
            size, in_grad->data_ptr<spec_t>());
        }
        else {
          sigmoid_grad_cpu<spec_t>(
            out_grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
            size, in_grad->data_ptr<spec_t>(), out_grad->ndim(),
            output->stride().data(), out_grad->stride().data(),
            in_grad->stride().data(), output->shape().data());
        }
        },"SigmoidGradient");  
    });
  NDArray::MarkUsedBy({output, out_grad, in_grad}, stream);
}

} // namespace impl
} // namespace hetu

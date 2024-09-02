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
void tanh_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::tanh(input[idx]);
  }
}

template <typename spec_t>
void tanh_cpu(const spec_t* input, size_t size, spec_t* output,
              int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = std::tanh(input[i_idx]);
  }
}

template <typename spec_t>
void tanh_cpu(const spec_t* input, size_t size, spec_t* output,
              int64_t ndims, const int64_t* stride, const int64_t* stride_out,
              const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = std::tanh(input[i_idx]);
  }
}

template <typename spec_t>
void tanh_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = (1 - input[idx] * input[idx]) * output_grad[idx];
  }
}

template <typename spec_t>
void tanh_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                       size_t size, spec_t* output, 
                       int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                       const int64_t* stride_in, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t og_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    int64_t ig_idx = hetu::impl::get_index(idx, ndims, stride_in, c_shape);
    output[idx] = (1 - input[idx] * input[idx]) * output_grad[idx];
  }
}

void TanhCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TanhCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output, size]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, 0);
          auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
          auto mat_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
          auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

          auto Tanh_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_tanh, mat_md, mat_md, float(0.0), float(0.0));
          auto Tanh = dnnl::eltwise_forward(Tanh_pd);

          std::unordered_map<int, dnnl::memory> tanh_args;
          tanh_args.insert({DNNL_ARG_SRC, src_mem});
          tanh_args.insert({DNNL_ARG_DST, dst_mem});      

          dnnl::stream engine_stream(eng);
          Tanh.execute(engine_stream, tanh_args);
          engine_stream.wait();
          engine_stream.wait();
        },"Tanh");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void TanhGradientCpu(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TanhGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output_grad, input_grad, size]() {
        if (input->is_contiguous() && output_grad->is_contiguous() && input_grad->is_contiguous()) {
          tanh_gradient_cpu<spec_t>(input->data_ptr<spec_t>(),
                                    output_grad->data_ptr<spec_t>(), size,
                                    input_grad->data_ptr<spec_t>());
        }
        else {
          tanh_gradient_cpu<spec_t>(input->data_ptr<spec_t>(),
                                    output_grad->data_ptr<spec_t>(), size,
                                    input_grad->data_ptr<spec_t>(), input->ndim(),
                                    input->stride().data(), output_grad->stride().data(),
                                    input_grad->stride().data(), input->shape().data());
        } 
        },"TanhGradient");
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu

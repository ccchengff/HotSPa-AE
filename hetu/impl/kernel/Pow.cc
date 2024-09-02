#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "cmath"

namespace hetu {
namespace impl {

template <typename spec_t>
void pow_cpu(const spec_t* input, double exponent, size_t size,
             spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::pow(input[idx], exponent);
  }
}

template <typename spec_t>
void pow_cpu(const spec_t* input, double exponent, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = std::pow(input[i_idx], exponent);
  }
}

template <typename spec_t>
void pow_cpu(const spec_t* input, double exponent, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* stride_out,
             const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = std::pow(input[i_idx], exponent);
  }
}

void PowCpu(const NDArray& input, double exponent, NDArray& output,
            const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "PowCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output, exponent]() {
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
        auto mat_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
        auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
        auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

        auto Pow_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                      dnnl::algorithm::eltwise_pow, mat_md, mat_md, float(1.0), float(exponent));
        auto Pow = dnnl::eltwise_forward(Pow_pd);

        std::unordered_map<int, dnnl::memory> pow_args;
        pow_args.insert({DNNL_ARG_SRC, src_mem});
        pow_args.insert({DNNL_ARG_DST, dst_mem});      

        dnnl::stream engine_stream(eng);
        Pow.execute(engine_stream, pow_args);
        engine_stream.wait();
      },"Pow");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu

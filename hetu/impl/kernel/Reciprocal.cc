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
void reciprocal_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = static_cast<spec_t>(1) / input[idx];
}

void ReciprocalCpu(const NDArray& input, NDArray& output,
                   const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_CPU_DEVICE(output);
  HT_ASSERT_EXCHANGABLE(input, output);
  size_t size = input->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ReciprocalCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output]() {
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
        auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
        auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
        auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

        auto Reciprocal_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                            dnnl::algorithm::eltwise_pow, mat_md, mat_md, float(1.0), float(-1.f));
        auto Reciprocal = dnnl::eltwise_forward(Reciprocal_pd);

        std::unordered_map<int, dnnl::memory> reciprocal_args;
        reciprocal_args.insert({DNNL_ARG_SRC, src_mem});
        reciprocal_args.insert({DNNL_ARG_DST, dst_mem});      

          dnnl::stream engine_stream(eng);
          Reciprocal.execute(engine_stream, reciprocal_args);
          engine_stream.wait();
        },"Reciprocal");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu

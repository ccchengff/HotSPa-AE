#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

void MatMulCpu(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
               NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  int32_t m = output->shape(0);
  int32_t n = output->shape(1);
  int32_t k = trans_a ? a->shape(0) : a->shape(1);

  CPUStream cpu_stream(stream);
  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatMul", [&]() {
    auto _future = cpu_stream.EnqueueTask(
    [stream, a, b, trans_a, trans_b, output, m, n, k]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      dnnl::memory::desc srcA_md, srcB_md, dst_md;
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(output->dtype());
      if (!trans_a)
          srcA_md = dnnl::memory::desc({m, k}, dnnltype, 
                                        dnnl::memory::format_tag::ab);
      else
          srcA_md = dnnl::memory::desc({m, k}, dnnltype, 
                                        dnnl::memory::format_tag::ba);
      if (!trans_b)
          srcB_md = dnnl::memory::desc({k, n}, dnnltype, 
                                        dnnl::memory::format_tag::ab);
      else
          srcB_md = dnnl::memory::desc({k, n}, dnnltype, 
                                        dnnl::memory::format_tag::ba);
      dst_md = dnnl::memory::desc({m, n}, dnnltype, 
                                  dnnl::memory::format_tag::ab);
                          
      auto srcA_mem = dnnl::memory(srcA_md, eng, a->data_ptr<spec_t>());
      auto srcB_mem = dnnl::memory(srcB_md, eng, b->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

      auto Matmul_pd = dnnl::matmul::primitive_desc(eng, srcA_md, srcB_md, dst_md);
      auto Matmul = dnnl::matmul(Matmul_pd);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, srcA_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, srcB_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});

      dnnl::stream engine_stream(eng);
      Matmul.execute(engine_stream, matmul_args);
      engine_stream.wait();
    },"Matmul");
  });
  NDArray::MarkUsedBy({a, b, output}, stream);
}

} // namespace impl
} // namespace hetu

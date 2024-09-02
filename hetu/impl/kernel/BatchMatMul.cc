#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

void BatchMatMulCpu(const NDArray& a, bool trans_a, const NDArray& b,
                    bool trans_b, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  int ndim = a->ndim();
  int m = output->shape(ndim - 2);
  int n = output->shape(ndim - 1);
  int k = trans_a ? a->shape(ndim - 2) : a->shape(ndim - 1);
  int batchCount = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    HT_ASSERT(a->shape(i) == b->shape(i));
    HT_ASSERT(a->shape(i) == output->shape(i));
    batchCount *= a->shape(i);
  }

  CPUStream cpu_stream(stream);
  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "BatchMatMul", [&]() {
    auto _future = cpu_stream.EnqueueTask(
    [stream, a, b, trans_a, trans_b, output, m, n, k, batchCount]() {
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(output->dtype());
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      dnnl::memory::desc srcA_md, srcB_md, dst_md;
      if (!trans_a)
          srcA_md = dnnl::memory::desc({batchCount, m, k}, dnnltype, 
                                        dnnl::memory::format_tag::abc);
      else
          srcA_md = dnnl::memory::desc({batchCount, m, k}, dnnltype, 
                                        dnnl::memory::format_tag::acb);
      if (!trans_b)
          srcB_md = dnnl::memory::desc({batchCount, k, n}, dnnltype, 
                                        dnnl::memory::format_tag::abc);
      else
          srcB_md = dnnl::memory::desc({batchCount, k, n}, dnnltype, 
                                        dnnl::memory::format_tag::acb);
      dst_md = dnnl::memory::desc({batchCount, m, n}, dnnltype, 
                                  dnnl::memory::format_tag::abc);
                          
      auto srcA_mem = dnnl::memory(srcA_md, eng, a->data_ptr<spec_t>());
      auto srcB_mem = dnnl::memory(srcB_md, eng, b->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

      auto Matmul_pd = dnnl::matmul::primitive_desc(eng, srcA_md, srcB_md, dst_md);
      auto Matmul = dnnl::matmul(Matmul_pd);

      std::unordered_map<int, dnnl::memory> bmm_args;
      bmm_args.insert({DNNL_ARG_SRC, srcA_mem});
      bmm_args.insert({DNNL_ARG_WEIGHTS, srcB_mem});
      bmm_args.insert({DNNL_ARG_DST, dst_mem});

      dnnl::stream engine_stream(eng);
      Matmul.execute(engine_stream, bmm_args);
      engine_stream.wait();
    },
    "BatchMatmul");
  });
  NDArray::MarkUsedBy({a, b, output}, stream);
}

} // namespace impl
} // namespace hetu

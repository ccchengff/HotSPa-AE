#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

void ReduceCpu(const NDArray& input, NDArray& output, const HTAxes& axes,
                ReductionType red_type, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  CPUStream cpu_stream(stream);
  HTAxes parsed_axes = NDArrayMeta::ParseAxes(axes, input->ndim());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "ReduceCpu", [&]() {
    dnnl::memory::dims in_shape = input->shape();
    dnnl::memory::dims in_stride = input->stride();
    dnnl::memory::dims out_shape = input->shape();
    dnnl::memory::dims out_stride(input->ndim());
    for (size_t i = 0; i < parsed_axes.size(); ++i) {
      out_shape[parsed_axes[i]] = 1;
    }
    size_t stride_size = 1;
    for (int i = input->ndim() - 1; i >= 0; i--) {
      out_stride[i] = stride_size;
      stride_size *= out_shape[i];
    }
    auto _future = cpu_stream.EnqueueTask(
      [input, output, in_shape, in_stride, out_shape, out_stride, red_type]() {
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
        auto src_md = dnnl::memory::desc(in_shape, dnnltype, in_stride);
        auto dst_md = dnnl::memory::desc(out_shape, dnnltype, out_stride);

        auto src_mem = dnnl::memory(src_md, eng, input->data_ptr<spec_t>());
        auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

        dnnl::algorithm algo = dnnl::algorithm::reduction_mean;
        if (red_type == ReductionType::MEAN)
          algo = dnnl::algorithm::reduction_mean;
        else if (red_type == ReductionType::SUM)
          algo = dnnl::algorithm::reduction_sum;
        else
          HT_NOT_IMPLEMENTED << "Invalid reduction type.";        

        if (in_shape == out_shape) {
          hetu::cpu::read_from_dnnl_memory(output->data_ptr<spec_t>(), src_mem);
        }
        else {
          // Create primitive descriptor.
          auto reduction_pd = dnnl::reduction::primitive_desc(
                  eng, algo, src_md, dst_md, float(0.f), float(0.f));

          // Create the primitive.
          auto reduction_prim = dnnl::reduction(reduction_pd);

          // Primitive arguments.
          std::unordered_map<int, dnnl::memory> reduction_args;
          reduction_args.insert({DNNL_ARG_SRC, src_mem});
          reduction_args.insert({DNNL_ARG_DST, dst_mem});

          // Primitive execution: Reduction (Sum).
          dnnl::stream engine_stream(eng);
          reduction_prim.execute(engine_stream, reduction_args);
          engine_stream.wait();
        } 
      },"Reduce"); 
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

void ReduceMinCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                  int64_t num_axes, const Stream& stream) {
}

void ReduceMaxCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                  int64_t num_axes, const Stream& stream) { 
}

void ReduceMeanCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                   int64_t num_axes, const Stream& stream) {
}

void ReduceSumCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                  int64_t num_axes, const Stream& stream) {
}

} // namespace impl
} // namespace hetu

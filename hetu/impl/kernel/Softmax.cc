#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

void SoftmaxCpu(const NDArray& input, NDArray& output, int64_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output, dim]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, 0);
          auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
          auto src_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
          auto dst_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
          auto src_mem = dnnl::memory(src_md, eng, input->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

          // Softmax axis.
          const int axis = dim >= 0 ? dim : dim + input->ndim();

          // Create primitive descriptor.
          auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                            dnnl::prop_kind::forward_training, 
                            dnnl::algorithm::softmax_accurate, 
                            src_md, dst_md, axis);

          // Create the primitive.
          auto softmax_prim = dnnl::softmax_forward(softmax_pd);

          // Primitive arguments. Set up in-place execution by assigning src as DST.
          std::unordered_map<int, dnnl::memory> softmax_args;
          softmax_args.insert({DNNL_ARG_SRC, src_mem});
          softmax_args.insert({DNNL_ARG_DST, dst_mem});

          dnnl::stream engine_stream(eng);
          softmax_prim.execute(engine_stream, softmax_args);
          engine_stream.wait();
        },"Softmax");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SoftmaxGradientCpu(const NDArray& input_Y, const NDArray& output_grad,
                        NDArray& input_grad, int64_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_Y);
  HT_ASSERT_SAME_DEVICE(input_Y, output_grad);
  HT_ASSERT_SAME_DEVICE(input_Y, input_grad);

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_Y->dtype(), spec_t, "SoftmaxGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input_Y, output_grad, input_grad, dim]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, 0);
          auto dnnltype = hetu::cpu::dtype_to_dnnltype(input_Y->dtype());
          auto src_md = dnnl::memory::desc(input_Y->shape(), dnnltype, input_Y->stride());
          auto dst_md = dnnl::memory::desc(input_Y->shape(), dnnltype, input_Y->stride());
          auto dst_mem = dnnl::memory(dst_md, eng, input_Y->data_ptr<spec_t>());
          auto gdst_mem = dnnl::memory(dst_md, eng, output_grad->data_ptr<spec_t>());
          auto gsrc_mem = dnnl::memory(src_md, eng, input_grad->data_ptr<spec_t>());

          // Softmax axis.
          const int axis = dim;

          // Create primitive descriptor.
          auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                            dnnl::prop_kind::forward_training, 
                            dnnl::algorithm::softmax_accurate, 
                            src_md, dst_md, axis);
          
          auto softmax_bwd_pd = dnnl::softmax_backward::primitive_desc(eng, dnnl::algorithm::softmax_accurate, 
                                                                      src_md, dst_md, dst_md, axis, softmax_pd);

          // Create the primitive.
          auto softmax_prim = dnnl::softmax_backward(softmax_bwd_pd);

          // Primitive arguments. Set up in-place execution by assigning src as DST.
          std::unordered_map<int, dnnl::memory> softmax_args;
          softmax_args.insert({DNNL_ARG_DIFF_SRC, gsrc_mem});
          softmax_args.insert({DNNL_ARG_DIFF_DST, gdst_mem});
          softmax_args.insert({DNNL_ARG_DST, dst_mem});
          dnnl::stream engine_stream(eng);
          softmax_prim.execute(engine_stream, softmax_args);
          engine_stream.wait();
        },"SoftmaxGradient");
    });
  NDArray::MarkUsedBy({input_Y, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu

#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void avgpool_cpu(const size_t threads, const spec_t* input_data,
                 spec_t* output_data, const size_t N, const size_t C,
                 const size_t H, const size_t W, const size_t kernel_H,
                 const size_t kernel_W, const size_t p_H, const size_t p_W,
                 const size_t padding, const size_t stride) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < threads; ++idx) {
    size_t idx_W = idx % p_W;
    idx /= p_W;
    size_t idx_H = idx % p_H;
    idx /= p_H;
    size_t idx_C = idx % C;
    size_t idx_N = idx / C;
    int hs = (int) idx_H * stride - padding;
    int ws = (int) idx_W * stride - padding;
    size_t hend = std::min(hs + kernel_H, H);
    size_t wend = std::min(ws + kernel_W, W);
    hs = std::max(hs, 0);
    ws = std::max(ws, 0);
    float temp = 0;
    for (size_t i = hs; i < hend; i++) {
      for (size_t j = ws; j < wend; j++) {
        temp += input_data[idx_N * C * H * W + idx_C * H * W + i * W + j];
      }
    }
    output_data[idx] = temp / (kernel_H * kernel_W);
  }
}

template <typename spec_t>
void avgpool_gradient_cpu(const size_t threads, const spec_t* input_data,
                          spec_t* output_data, const size_t N, const size_t C,
                          const size_t H, const size_t W, const size_t kernel_H,
                          const size_t kernel_W, const size_t p_H,
                          const size_t p_W, const size_t padding,
                          const size_t stride) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < threads; ++idx) {
    size_t idx_W = idx % p_W;
    idx /= p_W;
    size_t idx_H = idx % p_H;
    idx /= p_H;
    size_t idx_C = idx % C;
    size_t idx_N = idx / C;
    size_t hs = (idx_H < kernel_H) ? 0 : (idx_H - kernel_H) / stride + 1;
    size_t hend = std::min(idx_H / stride + 1, H);
    size_t ws = (idx_W < kernel_W) ? 0 : (idx_W - kernel_W) / stride + 1;
    size_t wend = std::min(idx_W / stride + 1, W);
    float temp = 0;
    const size_t pooling_size = kernel_H * kernel_W;
    for (size_t i = hs; i < hend; i++) {
      for (size_t j = ws; j < wend; j++) {
        temp += input_data[idx_N * C * H * W + idx_C * H * W + i * W + j];
      }
    }
    output_data[idx] = temp / pooling_size;
  }
}

void AvgPoolCpu(const NDArray& input, const size_t kernel_H,
                const size_t kernel_W, NDArray& output, const size_t padding,
                const size_t stride, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t input_N = input->shape(0);
  size_t input_C = input->shape(1);
  size_t output_H = output->shape(2);
  size_t output_W = output->shape(3);
  size_t output_size = input_N * input_C * output_H * output_W;

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  
  if (output_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "AvgPoolCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [eng, input, output, kernel_H, kernel_W,
        padding, stride]() {
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
        auto src_md = dnnl::memory::desc(input->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto src_mem = dnnl::memory(src_md, eng, input->data_ptr<spec_t>());

        auto dst_md = dnnl::memory::desc(output->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

        // Create primitive descriptor.
        dnnl::memory::dims strides_dims = {int(stride), int(stride)};
        dnnl::memory::dims kernel_dims = {int(kernel_H), int(kernel_W)};
        dnnl::memory::dims dilation = {0, 0};
        dnnl::memory::dims padding_dims_l = {int(padding), int(padding)};
        dnnl::memory::dims padding_dims_r = {int(padding), int(padding)};
        auto pooling_pd = dnnl::pooling_forward::primitive_desc(eng,
                dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_avg_include_padding, 
                src_md, dst_md, strides_dims, kernel_dims, dilation, padding_dims_l, padding_dims_r);

        auto workspace_mem = dnnl::memory(pooling_pd.workspace_desc(), eng);

        auto pooling_prim = dnnl::pooling_forward(pooling_pd);

        std::unordered_map<int, dnnl::memory> pooling_args;
        pooling_args.insert({DNNL_ARG_SRC, src_mem});
        pooling_args.insert({DNNL_ARG_DST, dst_mem});
        pooling_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

        dnnl::stream engine_stream(eng);
        pooling_prim.execute(engine_stream, pooling_args);
        engine_stream.wait();
      },
      "AvgPool");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void AvgPoolGradientCpu(const NDArray& output_Y, const NDArray& gradient_Y,
                        const NDArray& input_X, const size_t kernel_H,
                        const size_t kernel_W, NDArray& gradient_X,
                        const size_t padding, const size_t stride,
                        const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_Y);
  HT_ASSERT_SAME_DEVICE(output_Y, gradient_Y);
  HT_ASSERT_SAME_DEVICE(output_Y, input_X);
  HT_ASSERT_SAME_DEVICE(output_Y, gradient_X);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "AvgPoolGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [eng, output_Y, gradient_Y,
        input_X, gradient_X, kernel_H, kernel_W,
        padding, stride]() {
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input_X->dtype());
        auto src_md = dnnl::memory::desc(input_X->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto src_mem = dnnl::memory(src_md, eng, input_X->data_ptr<spec_t>());

        auto dst_md = dnnl::memory::desc(output_Y->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto dst_mem = dnnl::memory(dst_md, eng, output_Y->data_ptr<spec_t>());

        auto gdst_md = dnnl::memory::desc(gradient_Y->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto gdst_mem = dnnl::memory(gdst_md, eng, gradient_Y->data_ptr<spec_t>());
      
        auto gsrc_md = dnnl::memory::desc(gradient_X->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto gsrc_mem = dnnl::memory(gsrc_md, eng, gradient_X->data_ptr<spec_t>());

        // Create primitive descriptor.
        dnnl::memory::dims strides_dims = {int(stride), int(stride)};
        dnnl::memory::dims kernel_dims = {int(kernel_H), int(kernel_W)};
        dnnl::memory::dims dilation = {0, 0};
        dnnl::memory::dims padding_dims_l = {int(padding), int(padding)};
        dnnl::memory::dims padding_dims_r = {int(padding), int(padding)};
        auto pooling_pd = dnnl::pooling_forward::primitive_desc(eng,
                dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_avg_include_padding, 
                src_md, dst_md, strides_dims, kernel_dims, dilation, padding_dims_l, padding_dims_r);
        
        auto pooling_bwd_pd = dnnl::pooling_backward::primitive_desc(eng,
                dnnl::algorithm::pooling_avg_include_padding, 
                gsrc_md, gdst_md, strides_dims, kernel_dims, dilation, 
                padding_dims_l, padding_dims_r, pooling_pd);

        auto workspace_mem = dnnl::memory(pooling_pd.workspace_desc(), eng);

        auto pooling_prim = dnnl::pooling_backward(pooling_bwd_pd);

        std::unordered_map<int, dnnl::memory> pooling_args;
        pooling_args.insert({DNNL_ARG_SRC, src_mem});
        pooling_args.insert({DNNL_ARG_DST, dst_mem});
        pooling_args.insert({DNNL_ARG_DIFF_SRC, gsrc_mem});
        pooling_args.insert({DNNL_ARG_DIFF_DST, gdst_mem});
        pooling_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

        dnnl::stream engine_stream(eng);
        pooling_prim.execute(engine_stream, pooling_args);
        engine_stream.wait();
      },
      "AvgPoolGradient");     
    });
  NDArray::MarkUsedBy({output_Y, gradient_Y, input_X, gradient_X}, stream);
}

} // namespace impl
} // namespace hetu

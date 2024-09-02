#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {
void BatchNormCpu(const NDArray& input_X, const NDArray& bn_scale,
                  const NDArray& bn_bias, NDArray& output_Y, double momentum,
                  double eps, NDArray& running_mean, NDArray& running_var,
                  NDArray& save_mean, NDArray& save_var,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_X);
  HT_ASSERT_SAME_DEVICE(input_X, bn_scale);
  HT_ASSERT_SAME_DEVICE(input_X, bn_bias);
  HT_ASSERT_SAME_DEVICE(input_X, output_Y);
  HT_ASSERT_SAME_DEVICE(input_X, running_mean);
  HT_ASSERT_SAME_DEVICE(input_X, running_var);
  HT_ASSERT_SAME_DEVICE(input_X, save_mean);
  HT_ASSERT_SAME_DEVICE(input_X, save_var);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "BatchNormCuda", [&]() {
        auto _future = cpu_stream.EnqueueTask(
        [eng, input_X, bn_scale, bn_bias,
         output_Y, save_mean, save_var, momentum, eps]() {
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input_X->dtype());
        auto src_md = dnnl::memory::desc(input_X->shape(), dnnltype, input_X->stride());
        auto dst_md = dnnl::memory::desc(output_Y->shape(), dnnltype, output_Y->stride());
        auto scaleshift_md = dnnl::memory::desc(bn_bias->shape(), dnnltype, dnnl::memory::format_tag::x);

        auto src_mem = dnnl::memory(src_md, eng, input_X->data_ptr<spec_t>());
        auto dst_mem = dnnl::memory(dst_md, eng, output_Y->data_ptr<spec_t>());
        auto scale_mem = dnnl::memory(scaleshift_md, eng, bn_scale->data_ptr<spec_t>());
        auto shift_mem = dnnl::memory(scaleshift_md, eng, bn_bias->data_ptr<spec_t>());

        auto bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng,
                dnnl::prop_kind::forward_training, src_md, dst_md, float(eps),
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

        auto mean_mem = dnnl::memory(bnorm_pd.mean_desc(), eng, save_mean->data_ptr<spec_t>());
        auto variance_mem = dnnl::memory(bnorm_pd.variance_desc(), eng, save_var->data_ptr<spec_t>());
        auto workspace_mem = dnnl::memory(bnorm_pd.workspace_desc(), eng);

        auto bnorm_prim = dnnl::batch_normalization_forward(bnorm_pd);

        std::unordered_map<int, dnnl::memory> bnorm_args;
        bnorm_args.insert({DNNL_ARG_SRC, src_mem});
        bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
        bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
        bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
        bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
        bnorm_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
        bnorm_args.insert({DNNL_ARG_DST, dst_mem});

        dnnl::stream engine_stream(eng);
        bnorm_prim.execute(engine_stream, bnorm_args);
        engine_stream.wait();
      },
      "BatchNorm");
      
    });
  NDArray::MarkUsedBy({input_X, bn_scale, bn_bias, output_Y,
                       running_mean, running_var, save_mean, save_var}, stream);
}

void BatchNormGradientCpu(const NDArray& gradient_Y, const NDArray& input_X,
                          const NDArray& bn_scale, NDArray& gradient_X,
                          NDArray& gradient_bn_scale,
                          NDArray& gradient_bn_bias, double eps,
                          NDArray& save_mean, NDArray& save_var,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(gradient_Y);
  HT_ASSERT_SAME_DEVICE(gradient_Y, input_X);
  HT_ASSERT_SAME_DEVICE(gradient_Y, bn_scale);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_X);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_bn_scale);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_bn_bias);
  HT_ASSERT_SAME_DEVICE(gradient_Y, save_mean);
  HT_ASSERT_SAME_DEVICE(gradient_Y, save_var);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "BatchNormGradientCpu", [&]() {
        auto _future = cpu_stream.EnqueueTask(
        [eng, gradient_Y, input_X, bn_scale, gradient_X,
         gradient_bn_scale, gradient_bn_bias, save_mean, save_var, eps]() {
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input_X->dtype());
        auto src_md = dnnl::memory::desc(input_X->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto gdst_md = dnnl::memory::desc(gradient_Y->shape(), dnnltype, dnnl::memory::format_tag::nchw);
        auto scaleshift_md = dnnl::memory::desc(bn_scale->shape(), dnnltype, dnnl::memory::format_tag::x);
        auto mean_md = dnnl::memory::desc(save_mean->shape(), dnnltype, save_mean->stride());


        auto src_mem = dnnl::memory(src_md, eng, input_X->data_ptr<spec_t>());
        auto gsrc_mem = dnnl::memory(src_md, eng, gradient_X->data_ptr<spec_t>());
        auto gdst_mem = dnnl::memory(gdst_md, eng, gradient_Y->data_ptr<spec_t>());
        auto mean_mem = dnnl::memory(mean_md, eng, save_mean->data_ptr<spec_t>());
        auto variance_mem = dnnl::memory(mean_md, eng, save_var->data_ptr<spec_t>());
        auto scale_mem = dnnl::memory(scaleshift_md, eng, bn_scale->data_ptr<spec_t>());
        auto gscale_mem = dnnl::memory(scaleshift_md, eng, gradient_bn_scale->data_ptr<spec_t>());
        auto gbias_mem = dnnl::memory(scaleshift_md, eng, gradient_bn_bias->data_ptr<spec_t>());

        // Create primitive descriptor.
        auto bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng,
                dnnl::prop_kind::forward_training, src_md, gdst_md, float(eps),
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

        auto bnorm_bwd_pd = dnnl::batch_normalization_backward::primitive_desc(eng,
                dnnl::prop_kind::backward, src_md, gdst_md, src_md, float(eps),
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift, bnorm_pd);
        
        auto workspace_mem = dnnl::memory(bnorm_bwd_pd.workspace_desc(), eng);

        auto bnorm_prim = dnnl::batch_normalization_backward(bnorm_bwd_pd);

        std::unordered_map<int, dnnl::memory> bnorm_args;
        bnorm_args.insert({DNNL_ARG_SRC, src_mem});
        bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
        bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
        bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
        bnorm_args.insert({DNNL_ARG_DIFF_SCALE, gscale_mem});
        bnorm_args.insert({DNNL_ARG_DIFF_SHIFT, gbias_mem});
        bnorm_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
        bnorm_args.insert({DNNL_ARG_DIFF_DST, gdst_mem});
        bnorm_args.insert({DNNL_ARG_DIFF_SRC, gsrc_mem});

        dnnl::stream engine_stream(eng);
        bnorm_prim.execute(engine_stream, bnorm_args);
        engine_stream.wait();
      },
         "BatchNormGradient");
    });
  NDArray::MarkUsedBy({gradient_Y, input_X, bn_scale, gradient_X,
                       gradient_bn_scale, gradient_bn_bias, save_mean, save_var}, stream);
}    
} // namespace impl
} // namespace hetu

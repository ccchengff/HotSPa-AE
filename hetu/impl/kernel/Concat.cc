#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void Concat_cpu(const spec_t* inputA, const spec_t* inputB, size_t size,
                size_t concat_size, size_t offset1, size_t offset2,
                spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int all_offset = offset1 + offset2;
    int post_ind = idx % concat_size;
    int temp = idx / concat_size;
    int mid_ind = temp % all_offset;
    int pre_ind = temp / all_offset;
    float val;
    if (mid_ind < (int) offset1) {
      int x_ind = (pre_ind * offset1 + mid_ind) * concat_size + post_ind;
      val = inputA[x_ind];
    } else {
      int y_ind =
        (pre_ind * offset2 + mid_ind - offset1) * concat_size + post_ind;
      val = inputB[y_ind];
    }
    output[idx] = val;
  }
}

template <typename spec_t>
void Concat_gradient_cpu(const spec_t* output_grad, size_t size,
                         int concat_size, int concat_offset, int small_offset,
                         int big_offset, spec_t* input_grad) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int temp = idx / concat_size;
    int mid_ind = temp % small_offset + concat_offset;
    int pre_ind = temp / small_offset;
    int o_idx = (pre_ind * big_offset + mid_ind) * concat_size + post_ind;
    input_grad[idx] = output_grad[o_idx];
  }
}

void ConcatCpu(const NDArray& inputA, const NDArray& inputB, NDArray& output,
               size_t axis, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, inputB);
  HT_ASSERT_SAME_DEVICE(inputA, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = output->numel();
  size_t offset1 = inputA->shape(axis);
  size_t offset2 = inputB->shape(axis);
  int concat_size = 1;
  int now_ndim = inputA->ndim();
  for (int i = axis + 1; i < now_ndim; i++) {
    int cur_dim = inputA->shape(i);
    concat_size *= cur_dim;
  }
  if (size == 0 || offset1 == 0 || offset2 == 0)
    return;

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "ConcatCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [inputA, inputB, output, axis, eng]() {
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(inputA->dtype());
        auto srcA_md = dnnl::memory::desc(inputA->shape(), dnnltype, inputA->stride());
        auto srcB_md = dnnl::memory::desc(inputB->shape(), dnnltype, inputB->stride());
        auto srcA_mem = dnnl::memory(srcA_md, eng, inputA->data_ptr<spec_t>());
        auto srcB_mem = dnnl::memory(srcB_md, eng, inputB->data_ptr<spec_t>());
      
        auto concat_pd = dnnl::concat::primitive_desc(eng, axis, {srcA_md, srcB_md});

        auto dst_mem = dnnl::memory(concat_pd.dst_desc(), eng, output->data_ptr<spec_t>());

        auto concat_prim = dnnl::concat(concat_pd);

        std::unordered_map<int, dnnl::memory> concat_args;
        concat_args.insert({DNNL_ARG_MULTIPLE_SRC, srcA_mem});
        concat_args.insert({DNNL_ARG_MULTIPLE_SRC + 1, srcB_mem});
        concat_args.insert({DNNL_ARG_DST, dst_mem});

        dnnl::stream engine_stream(eng);
        concat_prim.execute(engine_stream, concat_args);
        engine_stream.wait();
      },
      "Concat");
      
    });
  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

void ConcatGradientCpu(const NDArray& output_grad, NDArray& input_grad,
                       size_t axis, size_t id, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = input_grad->numel();
  size_t big_offset = output_grad->shape(axis);
  size_t small_offset = input_grad->shape(axis);
  size_t concat_offset = (id == 1) ? (big_offset - small_offset) : 0;
  size_t concat_size = 1;
  size_t now_ndim = output_grad->ndim();
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    concat_size *= cur_dim;
  }
  if (size == 0 || small_offset == 0 || big_offset == 0)
    return;

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [output_grad, input_grad, size, concat_size, concat_offset, small_offset, big_offset]() {
      Concat_gradient_cpu<spec_t>(output_grad->data_ptr<spec_t>(), size,
                                  concat_size, concat_offset, small_offset,
                                  big_offset, input_grad->data_ptr<spec_t>());
      },
      "ConcatGradient");   
    });
  NDArray::MarkUsedBy({output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu

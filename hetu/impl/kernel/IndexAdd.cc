#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void index_add_cpu(const spec_t* input, const spec_t* ids, size_t size,
                   size_t before_stride, size_t after_stride, size_t cur_stride,
                   spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    size_t b_index = idx / (cur_stride * after_stride);
    size_t p_index = idx % (cur_stride * after_stride);
    size_t c_index = p_index / after_stride;
    size_t a_index = p_index % after_stride;
    size_t id_num = ids[c_index];
    size_t i_index =
      b_index * (cur_stride * after_stride) + id_num * after_stride + a_index;
    output[idx] = input[i_index];
  }
}

void IndexAddCpu(const NDArray& input, const NDArray& id, NDArray& output,
                 size_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, id);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(id->ndim() == 1)
    << "invalid index shape.Expect dim=1, but get" << id->ndim();

  CPUStream cpu_stream(stream);

  size_t before_stride = 1;
  size_t after_stride = 1;
  size_t cur_stride = input->shape(dim);
  HT_ASSERT(id->numel() == cur_stride && input->shape() == output->shape())
    << "Invalid shapes.Index shape:" << id->shape()
    << "Input shape:" << input->shape() << "Output shape:" << output->shape();
  for (size_t i = 0; i < dim; ++i) {
    before_stride *= input->shape(i);
  }
  for (size_t i = dim + 1; i < input->ndim(); ++i) {
    after_stride *= input->shape(i);
  }
  size_t size = id->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "IndexAddCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, id, output, size, before_stride, after_stride, cur_stride]() {
      index_add_cpu<spec_t>(input->data_ptr<spec_t>(), id->data_ptr<spec_t>(),
                            size, before_stride, after_stride, cur_stride,
                            output->data_ptr<spec_t>());
      },"IndexAdd");   
    });
  NDArray::MarkUsedBy({input, id, output}, stream);
}

} // namespace impl
} // namespace hetu

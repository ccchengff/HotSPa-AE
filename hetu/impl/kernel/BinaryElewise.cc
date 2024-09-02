#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

void BinaryElewiseToolCpu(const NDArray& inputA, const NDArray& inputB,
                          NDArray& output, dnnl::algorithm op, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);
  CPUStream cpu_stream(stream);

  dnnl::memory::dims A_dims(output->ndim());
  dnnl::memory::dims A_stride(output->ndim());
  dnnl::memory::dims B_dims(output->ndim());
  dnnl::memory::dims B_stride(output->ndim());
  dnnl::memory::dims out_strides(output->ndim());

  size_t output_dim = output->ndim();
  size_t output_size = 1;
  size_t A_size = 1;
  size_t B_size = 1;
  size_t diff = output_dim - inputA->ndim();

  for (int i = output_dim - 1; i >= 0; --i) {
    out_strides[i] = output_size;
    output_size *= output->shape(i);
    if (i < int(diff)) {
      A_dims[i] = 1;
    } else {
      A_dims[i] = inputA->shape(i - diff);
    }
    A_stride[i] = A_size;
    A_size *= A_dims[i];
  }
  diff = output_dim - inputB->ndim();
  for (int i = output_dim - 1; i >= 0; --i) {
    if (i < int(diff)) {
      B_dims[i] = 1;
    } else {
      B_dims[i] = inputB->shape(i - diff);
    }
    B_stride[i] = B_size;
    B_size *= B_dims[i];
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "BinaryElewiseCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [inputA, inputB, output, A_dims, A_stride,
         B_dims, B_stride, out_strides, op]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, 0);
          auto dnnltype = hetu::cpu::dtype_to_dnnltype(inputA->dtype());
          auto src_A_md = dnnl::memory::desc(A_dims, dnnltype, A_stride);
          auto src_B_md = dnnl::memory::desc(B_dims, dnnltype, B_stride);
          auto dst_md = dnnl::memory::desc(output->shape(), dnnltype, out_strides);

          // Create src memory objects.
          auto src_A_mem = dnnl::memory(src_A_md, eng, inputA->data_ptr<spec_t>());
          auto src_B_mem = dnnl::memory(src_B_md, eng, inputB->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(dst_md, eng, output->data_ptr<spec_t>());

          auto binary_pd = dnnl::binary::primitive_desc(eng, op,
                  src_A_md, src_B_md, dst_md);

          // Create the primitive.
          auto binary_prim = dnnl::binary(binary_pd);

          // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
          std::unordered_map<int, dnnl::memory> binary_args;
          binary_args.insert({DNNL_ARG_SRC_0, src_A_mem});
          binary_args.insert({DNNL_ARG_SRC_1, src_B_mem});
          binary_args.insert({DNNL_ARG_DST, dst_mem});
          dnnl::stream engine_stream(eng);
          binary_prim.execute(engine_stream, binary_args);
          engine_stream.wait();
        },
        "BinaryEleWise");
    });
  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

void AddElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_add, stream);
}

void SubElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_sub, stream);
}

void MulElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_mul, stream);
}

void DivElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_div, stream); 
}

} // namespace impl
} // namespace hetu

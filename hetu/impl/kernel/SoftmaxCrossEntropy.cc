#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {
template <typename spec_t>
void softmax_cross_entropy_cpu(const spec_t* logsoftmax,
                               const spec_t* label,
                               spec_t* output, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) 
    output[idx] = -logsoftmax[idx] * label[idx];
}

void SoftmaxCrossEntropyCpu(const NDArray& input, const NDArray& label,
                            NDArray& output, const Stream& stream) {
  size_t indim = input->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (size_t i = 0; i < indim - 1; ++i) {
    n_ *= input->shape(i);
  }
  int c_ = input->shape(indim - 1);
  size_t size = n_ * c_;

  if (size == 0)
    return;

  auto workspace = NDArray::empty(input->shape(), input->device(), kFloat32,
                                  stream.stream_index());
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      cpu_stream.EnqueueTask(
        [input, label, output, workspace, stream, size]() {
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        void* workspace_ptr = workspace->raw_data_ptr();
        dnnl::stream engine_stream(eng);
        auto dnnltype = hetu::cpu::dtype_to_dnnltype(input->dtype());
        auto src_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
        auto dst_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
        auto src_mem = dnnl::memory(src_md, eng, input->data_ptr<spec_t>());
        auto dst_mem = dnnl::memory(dst_md, eng, workspace_ptr);

        // Softmax axis.
        const int axis = 1;
        auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                          dnnl::prop_kind::forward_training, 
                          dnnl::algorithm::softmax_log, 
                          src_md, dst_md, axis);

        auto softmax_prim = dnnl::softmax_forward(softmax_pd);

        std::unordered_map<int, dnnl::memory> softmax_args;
        softmax_args.insert({DNNL_ARG_SRC, src_mem});
        softmax_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution.
        softmax_prim.execute(engine_stream, softmax_args);
        engine_stream.wait();


        softmax_cross_entropy_cpu<spec_t>(
          (const spec_t*) workspace_ptr, label->data_ptr<spec_t>(),
          (spec_t*) workspace_ptr, size);

        HTShape outshape = output->shape(); outshape.emplace_back(1);
        HTShape outstride = output->stride(); outstride.emplace_back(1);
        auto rsrc_md = dnnl::memory::desc(input->shape(), dnnltype, input->stride());
        auto rdst_md = dnnl::memory::desc(outshape, dnnltype, outstride);

        auto rsrc_mem = dnnl::memory(rsrc_md, eng, workspace_ptr);
        auto rdst_mem = dnnl::memory(rdst_md, eng, output->data_ptr<spec_t>());

        if (input->shape() == outshape)
          hetu::cpu::read_from_dnnl_memory(output->data_ptr<spec_t>(), rsrc_mem);
        else {
          // Create primitive descriptor.
          auto reduction_pd = dnnl::reduction::primitive_desc(
                  eng, dnnl::algorithm::reduction_sum, rsrc_md, rdst_md, float(0.f), float(0.f));

          // Create the primitive.
          auto reduction_prim = dnnl::reduction(reduction_pd);

          // Primitive arguments.
          std::unordered_map<int, dnnl::memory> reduction_args;
          reduction_args.insert({DNNL_ARG_SRC, rsrc_mem});
          reduction_args.insert({DNNL_ARG_DST, rdst_mem});

          // Primitive execution: Reduction (Sum).
          reduction_prim.execute(engine_stream, reduction_args);
          engine_stream.wait();
        }
        },"SoftmaxCrossEntropy");
  });
  NDArray::MarkUsedBy({input, label, output, workspace}, stream);
} 

template <typename spec_t>
void softmax_cross_entropy_gradient_cpu(
  const spec_t* pred, const spec_t* y_, const spec_t* grad_data,
  spec_t* output_data, int last_dim, size_t size) {  
  for (size_t idx = 0; idx < size; ++idx)
    output_data[idx] = (pred[idx] - y_[idx]) * grad_data[idx / last_dim];
}

void SoftmaxCrossEntropyGradientCpu(const NDArray& input_y,
                                    const NDArray& label, const NDArray& grad,
                                    NDArray& output, const Stream& stream) {
  size_t indim = input_y->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() &&
            indim == grad->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (size_t i = 0; i < indim - 1; ++i) {
    n_ *= input_y->shape(i);
  }
  int c_ = input_y->shape(indim - 1);
  size_t size = n_ * c_;

  auto workspace = NDArray::empty(input_y->shape(), input_y->device(), kFloat32,
                                  stream.stream_index());
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_y->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input_y, label, grad, output, workspace, stream, c_, size]() {
        void* workspace_ptr = workspace->raw_data_ptr();
        
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(eng);
        
        auto src_md = dnnl::memory::desc(input_y->shape(), dnnl::memory::data_type::f32, input_y->stride());
        auto dst_md = dnnl::memory::desc(input_y->shape(), dnnl::memory::data_type::f32, input_y->stride());
        auto src_mem = dnnl::memory(src_md, eng, input_y->data_ptr<spec_t>());
        auto dst_mem = dnnl::memory(dst_md, eng, workspace_ptr);

        // Softmax axis.
        const int axis = 1;
        auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                          dnnl::prop_kind::forward_training, 
                          dnnl::algorithm::softmax_accurate, 
                          src_md, dst_md, axis);

        auto softmax_prim = dnnl::softmax_forward(softmax_pd);

        std::unordered_map<int, dnnl::memory> softmax_args;
        softmax_args.insert({DNNL_ARG_SRC, src_mem});
        softmax_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution.
        softmax_prim.execute(engine_stream, softmax_args);
        engine_stream.wait();

        softmax_cross_entropy_gradient_cpu<spec_t>(
            (const spec_t*) workspace_ptr, label->data_ptr<spec_t>(),
            grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(), c_, size);
        },"SoftmaxCrossEntropyGradient");
    });
  NDArray::MarkUsedBy({input_y, label, grad, output, workspace}, stream);
}
} // namespace impl
} // namespace hetu

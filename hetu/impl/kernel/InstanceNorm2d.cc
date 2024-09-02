#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void minus_mean_n_square_kernel1(const spec_t* in_arr,
                                 const spec_t* mean, spec_t* out_arr,
                                 int last_2dim, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    spec_t temp = in_arr[idx] - mean[idx / last_2dim];
    out_arr[idx] = temp * temp;
  }
}

template <typename spec_t>
void std_normal_transform(const spec_t* in_arr,
                          const spec_t* mean_arr,
                          const spec_t* var_arr, spec_t* out_arr,
                          int last_2dim, float eps, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t mo_idx = idx / last_2dim;
    out_arr[idx] =
      (in_arr[idx] - mean_arr[mo_idx]) / std::sqrt(var_arr[mo_idx] + eps);
  }
}

void InstanceNormCpu(const NDArray& in_arr, NDArray& mean_arr, NDArray& var_arr,
                     NDArray& out_arr, float eps, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormCpu", [&]() {
      cpu_stream.EnqueueTask(
      [stream, in_arr, mean_arr, var_arr, out_arr, eps, last_2dim, ndim]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      dnnl::stream engine_stream(eng); 
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(in_arr->dtype());
      auto src_md = dnnl::memory::desc(in_arr->shape(), dnnltype, in_arr->stride());
      auto dst_md = dnnl::memory::desc(mean_arr->shape(), dnnltype, mean_arr->stride());

      auto src_mem = dnnl::memory(src_md, eng, in_arr->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(dst_md, eng, mean_arr->data_ptr<spec_t>());

      if (in_arr->shape() == mean_arr->shape())
        hetu::cpu::read_from_dnnl_memory(mean_arr->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_mean, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      }

      engine_stream.wait();

      minus_mean_n_square_kernel1<spec_t>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(),
        out_arr->data_ptr<spec_t>(), last_2dim, in_arr->numel());

      // Write data to memory object's handle.
      src_mem = dnnl::memory(src_md, eng, out_arr->data_ptr<spec_t>());
      dst_mem = dnnl::memory(dst_md, eng, var_arr->data_ptr<spec_t>());
      if (in_arr->shape() == mean_arr->shape())
        hetu::cpu::read_from_dnnl_memory(var_arr->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_mean, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
        engine_stream.wait();
      }
      std_normal_transform<spec_t>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(),
        var_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), last_2dim,
        eps, in_arr->numel());    
      },"InstanceNorm");
      
    });
  NDArray::MarkUsedBy({in_arr, mean_arr, var_arr}, stream);
}

template <typename spec_t>
void calculate_grad_kernel(const spec_t* out_grads,
                           const spec_t* in_arr,
                           const spec_t* mean_arr,
                           const spec_t* var_arr, 
                           spec_t* ds, spec_t* dbias,
                           spec_t* grad_arr,
                           size_t last2dim, float eps, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t mo_idx = idx / last2dim;
    spec_t tmp = (dbias[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                  (var_arr[mo_idx] + eps);
    grad_arr[idx] = out_grads[idx] /std::sqrt(var_arr[mo_idx] + eps) +
      ((tmp - dbias[mo_idx]) / (spec_t)last2dim) / 
      std::sqrt(var_arr[mo_idx] + eps);
  }
}

void InstanceNormGradientCpu(const NDArray& out_grads, const NDArray& in_arr,
                             NDArray& grad_arr, const NDArray& mean_arr,
                             const NDArray& var_arr, float eps,
                             const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr);   
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 

  int ndim = out_grads->ndim();
  HT_ASSERT(ndim == 4);

  HT_ASSERT(ndim == 4);

  int last2dim = out_grads->shape(ndim - 1) * out_grads->shape(ndim - 2);

  size_t size = out_grads->numel();
  if (size == 0)
    return;
  
  auto dscale_arr = NDArray::empty_like(mean_arr, stream.stream_index());
  auto dbias_arr = NDArray::empty_like(mean_arr, stream.stream_index());
  auto dy_mul_x_arr = NDArray::empty_like(in_arr, stream.stream_index());
  
  CPUStream cpu_stream(stream);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormGradientCpu", [&]() {
      cpu_stream.EnqueueTask(
      [stream, out_grads, in_arr, grad_arr, mean_arr, var_arr, 
      dscale_arr, dbias_arr, dy_mul_x_arr, eps, ndim, last2dim, size]() {
      spec_t* dscale = dscale_arr->data_ptr<spec_t>();
      spec_t* dbias = dbias_arr->data_ptr<spec_t>();
      spec_t* dy_mul_x = dy_mul_x_arr->data_ptr<spec_t>();
      
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      dnnl::stream engine_stream(eng); 
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(in_arr->dtype());
      auto src_md = dnnl::memory::desc(in_arr->shape(), dnnltype, in_arr->stride());
      auto dst_md = dnnl::memory::desc(mean_arr->shape(), dnnltype, mean_arr->stride());

      auto src_mem = dnnl::memory(src_md, eng, out_grads->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(dst_md, eng, dbias);

      if (in_arr->shape() == mean_arr->shape())
        hetu::cpu::read_from_dnnl_memory(dbias, src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      } 
      engine_stream.wait();
      // Create src memory objects.
      auto src_A_mem = dnnl::memory(src_md, eng, out_grads->data_ptr<spec_t>());
      auto src_B_mem = dnnl::memory(src_md, eng, in_arr->data_ptr<spec_t>());
      auto dymulx_mem = dnnl::memory(src_md, eng, dy_mul_x);

      auto binary_pd = dnnl::binary::primitive_desc(eng, dnnl::algorithm::binary_mul,
                                                    src_md, src_md, src_md);

      // Create the primitive.
      auto binary_prim = dnnl::binary(binary_pd);

      // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC_0, src_A_mem});
      binary_args.insert({DNNL_ARG_SRC_1, src_B_mem});
      binary_args.insert({DNNL_ARG_DST, dymulx_mem});

      // Primitive execution: binary with ReLU.
      binary_prim.execute(engine_stream, binary_args);
      engine_stream.wait();

      dst_mem = dnnl::memory(dst_md, eng, dscale);
      
      if (in_arr->shape() == mean_arr->shape())
        hetu::cpu::read_from_dnnl_memory(dscale, dymulx_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, dst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, dymulx_mem});
        reduction_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      } 
      engine_stream.wait();
      calculate_grad_kernel<spec_t>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        dscale, dbias,
        grad_arr->data_ptr<spec_t>(), last2dim, eps, size);
      },"InstanceNormGradient");
    }); 
  NDArray::MarkUsedBy({out_grads, in_arr, mean_arr, var_arr, grad_arr}, stream);
}

} // namespace impl
} // namespace hetu

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
                                 int last_dims, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    spec_t temp = in_arr[idx] - mean[idx / last_dims];
    out_arr[idx] = temp * temp;
  }
}

template <typename spec_t>
void layer_norm_kernel(const spec_t* in_arr, const spec_t* mean_arr, const spec_t* var_arr, 
                       const spec_t* scale,const spec_t* bias, spec_t* out_arr,
                       int last_dims, float eps, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    size_t mo_idx = idx / last_dims;
    size_t add_idx = idx % last_dims;
    out_arr[idx] =
      (in_arr[idx] - mean_arr[mo_idx]) / std::sqrt(var_arr[mo_idx] + eps) * scale[add_idx] + bias[add_idx];
  }
}

void LayerNormCpu(const NDArray& in_arr, const NDArray& ln_scale,
                  const NDArray& ln_bias, NDArray& mean_arr, NDArray& var_arr,
                  NDArray& out_arr, int64_t reduce_dims, 
                  float eps, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_bias);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_dims = 1;

  HTShape dimA(ndim);
  HTShape strideA(ndim);
  HTShape dimC(ndim);
  HTShape strideC(ndim);

  int temp_strideA = 1;
  int temp_strideC = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    dimA[i] = (int) in_arr->shape(i);
    dimC[i] = i < int(in_arr->ndim() - reduce_dims) ? (int) in_arr->shape(i) : 1;
    if (i >= ndim - reduce_dims)
      last_dims *= in_arr->shape(i);
    strideA[i] = temp_strideA;
    strideC[i] = temp_strideC;
    temp_strideA *= dimA[i];
    temp_strideC *= dimC[i];
  }

  CPUStream cpu_stream(stream);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormCpu", [&]() {
      cpu_stream.EnqueueTask(
      [stream, in_arr, ln_scale, ln_bias, mean_arr, var_arr, out_arr, temp_strideA, temp_strideC,
       eps, last_dims, ndim]() {
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
        out_arr->data_ptr<spec_t>(), last_dims, temp_strideA);

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
      }
      engine_stream.wait();

      layer_norm_kernel<spec_t>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),  
        ln_scale->data_ptr<spec_t>(), ln_bias->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), 
        last_dims, eps, temp_strideA); 
      },"LayerNorm");
               
    });
  NDArray::MarkUsedBy({in_arr, ln_scale, ln_bias, mean_arr, var_arr, out_arr},
                      stream);
}

template <typename spec_t>
void calculate_gscale(const spec_t* grads, const spec_t* in_arr,
                      const spec_t* mean_arr, const spec_t* var_arr,
                      spec_t* grad_scale, spec_t eps,
                      int last_dim, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {
    int mo_ind = idx / last_dim;
    spec_t std = sqrtf(var_arr[mo_ind] + eps);
    spec_t x_centered = in_arr[idx] - mean_arr[mo_ind];
    spec_t x_norm = x_centered / std;
    grad_scale[idx] = grads[idx] * x_norm;
  }
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


void LayerNormGradientCpu(const NDArray& out_grads, const NDArray& in_arr,
                          const NDArray& ln_scale, NDArray& grad_arr,
                          NDArray& grad_scale, NDArray& grad_bias,
                          const NDArray& mean_arr, const NDArray& var_arr,
                          int64_t reduce_dims, float eps, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_bias);

  int ndim = out_grads->ndim();
  size_t total_elements = 1;

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int lastdims = 1;
  for (int i = 0; i < reduce_dims; ++i) {
    lastdims *= out_grads->shape(ndim - 1 -i);
  }

  size_t size = total_elements;
  if (size == 0)
    return;
  
  auto ds_arr = NDArray::empty_like(mean_arr, stream.stream_index());
  auto db_arr = NDArray::empty_like(mean_arr, stream.stream_index());
  auto dy_mul_x_arr = NDArray::empty_like(in_arr, stream.stream_index());
  auto gscale_arr = NDArray::empty_like(in_arr, stream.stream_index());

  CPUStream cpu_stream(stream);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormGradientCpu", [&]() {
      cpu_stream.EnqueueTask(
      [stream, out_grads, in_arr, ln_scale, grad_scale, grad_bias, grad_arr, mean_arr, var_arr, 
      ds_arr, db_arr, dy_mul_x_arr, gscale_arr,
      reduce_dims, eps, ndim, lastdims, total_elements, size]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      spec_t* ds = ds_arr->data_ptr<spec_t>();
      spec_t* db = db_arr->data_ptr<spec_t>();
      spec_t* dy_mul_x = dy_mul_x_arr->data_ptr<spec_t>();
      spec_t* gscale = gscale_arr->data_ptr<spec_t>();

      HTShape scale_shape(ndim), scale_stride(ndim);
      int64_t stride_size = 1;
      for (int i = 0; i < ndim; i++) {
        scale_shape[ndim - 1 - i] = (i < reduce_dims ? 
                                     ln_scale->shape(ln_scale->ndim() - 1 - i) : 1);
        scale_stride[ndim - 1 - i] = stride_size;
        stride_size *= scale_shape[ndim - 1 - i];
      }
      dnnl::stream engine_stream(eng);
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(in_arr->dtype());
      auto src_md = dnnl::memory::desc(in_arr->shape(), dnnltype, in_arr->stride());
      auto scale_md = dnnl::memory::desc(scale_shape, dnnltype, scale_stride);
      auto mean_md = dnnl::memory::desc(mean_arr->shape(), dnnltype, mean_arr->stride());

      auto src_mem = dnnl::memory(src_md, eng, out_grads->data_ptr<spec_t>());
      auto scale_mem = dnnl::memory(scale_md, eng, grad_bias->data_ptr<spec_t>());
      auto mean_mem = dnnl::memory(mean_md, eng);

      if (in_arr->shape() == ln_scale->shape())
        hetu::cpu::read_from_dnnl_memory(grad_bias->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, scale_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, scale_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      } 
      engine_stream.wait();

      calculate_gscale<spec_t>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        gscale, eps, lastdims, (size_t) in_arr->numel());
      
      src_mem = dnnl::memory(src_md, eng, gscale);
      scale_mem = dnnl::memory(scale_md, eng, grad_scale->data_ptr<spec_t>());
      if (in_arr->shape() == ln_scale->shape())
        hetu::cpu::read_from_dnnl_memory(grad_scale->data_ptr<spec_t>(), src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, scale_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, scale_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      } 

      src_mem = dnnl::memory(src_md, eng, out_grads->data_ptr<spec_t>());
      mean_mem = dnnl::memory(mean_md, eng, db);
      if (in_arr->shape() == mean_arr->shape())
        hetu::cpu::read_from_dnnl_memory(db, src_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, mean_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, src_mem});
        reduction_args.insert({DNNL_ARG_DST, mean_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      } 
      

      // Create src memory objects.
      auto src_A_mem = dnnl::memory(src_md, eng, out_grads->data_ptr<spec_t>());
      auto src_B_mem = dnnl::memory(src_md, eng, in_arr->data_ptr<spec_t>());
      auto mdst_mem = dnnl::memory(src_md, eng, dy_mul_x);

      auto binary_pd = dnnl::binary::primitive_desc(eng, dnnl::algorithm::binary_mul,
                                                    src_md, src_md, src_md);

      // Create the primitive.
      auto binary_prim = dnnl::binary(binary_pd);

      // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC_0, src_A_mem});
      binary_args.insert({DNNL_ARG_SRC_1, src_B_mem});
      binary_args.insert({DNNL_ARG_DST, mdst_mem});

      // Primitive execution: binary with ReLU.
      binary_prim.execute(engine_stream, binary_args);
      engine_stream.wait();

      mean_mem = dnnl::memory(mean_md, eng, ds);
      if (in_arr->shape() == ln_scale->shape())
        hetu::cpu::read_from_dnnl_memory(ds, mdst_mem);
      else {

        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, src_md, mean_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, mdst_mem});
        reduction_args.insert({DNNL_ARG_DST, mean_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);
      } 
      engine_stream.wait();

      calculate_grad_kernel<spec_t>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        ds, db,
        grad_arr->data_ptr<spec_t>(), lastdims, eps, size);
      },"LayerNormGradient");
    }); 
  NDArray::MarkUsedBy({out_grads, in_arr, ln_scale, grad_arr,
                       grad_scale, grad_bias, mean_arr, var_arr}, stream);
}


} // namespace impl
} // namespace hetu

#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/random/CPURandomState.h"
#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"
#include "layer_norm/ln.h"

using namespace layer_norm;

namespace hetu {
namespace impl {


uint32_t get_type_id(DataType dtype){
    if( dtype == kFloat16 ) {
        return TypeId<fp16>::Value;
    } else if( dtype == kBFloat16 ) {
        return TypeId<bf16>::Value;
    } else if( dtype == kFloat32 ) {
        return TypeId<fp32>::Value;
    } else {
        HT_NOT_IMPLEMENTED << "Type not supported: " << dtype;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(DataType wtype, DataType itype, DataType rtype, DataType otype, DataType ctype, uint64_t hidden_size) {
    // using namespace layer_norm;
    uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) | (get_type_id(rtype) << 4) | (get_type_id(otype) << 6) | (get_type_id(ctype) << 8);
    uint64_t launcher_key = (type_key << 32) | hidden_size;
    return launcher_key;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FwdFunction & get_fwd_launcher(DataType wtype, DataType itype, DataType rtype, DataType otype, DataType ctype, uint32_t hidden_size) {
    auto iter = FWD_FUNCS.find(get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    if( iter != FWD_FUNCS.end() ) {
        return iter->second;
    } else {
        HT_NOT_IMPLEMENTED << "FWD: Unsupported hidden_size or types: " 
        << hidden_size << "," << wtype << "," << itype << ","
        << rtype << "," << otype << "," << ctype;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BwdFunction & get_bwd_launcher(DataType wtype, DataType itype, DataType rtype, DataType otype, DataType ctype, uint32_t hidden_size) {
    auto iter = BWD_FUNCS.find(get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    if( iter != BWD_FUNCS.end() ) {
        return iter->second;
    } else {
        HT_NOT_IMPLEMENTED << "FWD: Unsupported hidden_size or types: " 
        << hidden_size << "," << wtype << "," << itype << ","
        << rtype << "," << otype << "," << ctype;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FwdFunction & get_parallel_fwd_launcher(DataType wtype, DataType itype, DataType rtype, DataType otype, DataType ctype, uint32_t hidden_size) {
    auto iter = PARALLEL_FWD_FUNCS.find(get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    if( iter != PARALLEL_FWD_FUNCS.end() ) {
        return iter->second;
    } else {
        HT_NOT_IMPLEMENTED << "FWD: Unsupported hidden_size or types: " 
        << hidden_size << "," << wtype << "," << itype << ","
        << rtype << "," << otype << "," << ctype;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BwdFunction & get_parallel_bwd_launcher(DataType wtype, DataType itype, DataType rtype, DataType otype, DataType ctype, uint32_t hidden_size) {
    auto iter = PARALLEL_BWD_FUNCS.find(get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    if( iter != PARALLEL_BWD_FUNCS.end() ) {
        return iter->second;
    } else {
        HT_NOT_IMPLEMENTED << "FWD: Unsupported hidden_size or types: " 
        << hidden_size << "," << wtype << "," << itype << ","
        << rtype << "," << otype << "," << ctype;
    }
}

void DropoutAddLnFwdCuda(const NDArray& x0,      // Input: BxSxhidden_size
                         const NDArray& residual_,  // Residual: BxSxhidden_size
                         const NDArray& gamma,   // hidden_size
                         const NDArray& beta_,   // hidden_size
                         const NDArray& rowscale_,      // BxS
                         const NDArray& colscale_,      // hidden_size
                         const NDArray& x0_subset_,      // BxS
                         const NDArray& z_subset_,      // BxS
                         NDArray& z, 
                         NDArray& x, 
                         NDArray& dmask, 
                         NDArray& mu, 
                         NDArray& rsigma,
                         const float dropout_p,
                         const float epsilon,
                         const float rowscale_const,
                         const int64_t z_numrows,
                         bool residual_in_fp32,
                         bool is_rms_norm,
                         const Stream& stream) {
  auto itype = x0->dtype();
  auto rtype = residual_.is_defined()
      ? residual_->dtype()
      : (residual_in_fp32 ? kFloat32 : x0->dtype());
  auto wtype = gamma->dtype();
  auto otype = itype;
  auto ctype = kFloat32;
  auto mtype = kUInt8;

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  HT_ASSERT(x0->is_cuda());
  HT_ASSERT(gamma->is_cuda());

  HT_ASSERT(x0->is_contiguous());
  //  does not own the storage, so we need to construct a vector.
  // Otherwise just constructing IntArrayRef({blah}) will cause uninitialized memory because
  // blah is then deallocated.
  HTShape sizes {!x0_subset_.is_defined() ? x0->shape(0) : x0_subset_->shape(0), x0->shape(1)};
  HT_ASSERT(x0->ndim() == 2);
  HT_ASSERT(sizes.size() == 2);

  const int rows = sizes[0];
  const int cols = sizes[1];
  auto hidden_size = gamma->numel();
  HT_ASSERT(hidden_size == cols);
  HTShape rows_shape{rows};
  HTShape cols_shape{cols};

  if (beta_.is_defined()) {
      auto beta = beta_;
      HT_ASSERT(beta->dtype() == wtype);
      HT_ASSERT(beta->is_cuda());
      HT_ASSERT(beta->is_contiguous());
      HT_ASSERT(beta->shape() == gamma->shape());
  }

  if (residual_.is_defined()) {
      auto residual = residual_;
      HT_ASSERT(residual->is_cuda());
      HT_ASSERT(residual->is_contiguous());
      HT_ASSERT(residual->shape() == sizes);
  }

  if (rowscale_.is_defined()) {
      auto rowscale = rowscale_;
      HT_ASSERT(rowscale->is_cuda());
      HT_ASSERT(rowscale->is_contiguous());
      HT_ASSERT(rowscale->shape() == rows_shape);
      HT_ASSERT(rowscale->dtype() == itype);
  }

  if (colscale_.is_defined()) {
      auto colscale = colscale_;
      HT_ASSERT(colscale->is_cuda());
      HT_ASSERT(colscale->is_contiguous());
      HT_ASSERT(colscale->shape() == cols_shape);
      HT_ASSERT(colscale->dtype() == wtype);
  }

  if (x0_subset_.is_defined()) {
      auto x0_subset = x0_subset_;
      HT_ASSERT(x0_subset->is_cuda());
      HT_ASSERT(x0_subset->is_contiguous());
      HT_ASSERT(x0_subset->shape() == rows_shape);
      HT_ASSERT(x0_subset->dtype() == kInt32);

      HT_ASSERT(z_subset_.is_defined());
      auto z_subset = z_subset_;
      HT_ASSERT(z_subset->is_cuda());
      HT_ASSERT(z_subset->is_contiguous());
      HT_ASSERT(z_subset->shape() == rows_shape);
      HT_ASSERT(z_subset->dtype() == kInt32);
  }

  HT_ASSERT((hidden_size % 8 == 0) && (hidden_size <= 8192));
  HT_ASSERT(epsilon >= 0.f);

  bool save_x = residual_.is_defined() || (dropout_p > 0.f) || rowscale_.is_defined() || colscale_.is_defined() || x0_subset_.is_defined() || (itype != rtype);
  // NDArray x;
  // if (save_x) { x = NDArray::empty(sizes, x0->device(), rtype, stream.stream_index()); }
  // NDArray dmask;
  // if (dropout_p > 0.f) { dmask = NDArray::empty(x0->shape(), x0->device(), mtype, stream.stream_index()); };
  // auto z = NDArray::empty(z_subset_.is_defined() ? {z_numrows, cols} : sizes, x0->device(), otype, stream.stream_index());

  // auto mu = NDArray::empty({ rows }, x0->device(), ctype, stream.stream_index());
  // auto rsigma = NDArray::empty({ rows }, x0->device(), ctype, stream.stream_index());

  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  cudaDeviceProp prop = Device::dprop(x0->device().index());
  launch_params.props = &prop;
  launch_params.stream = cuda_stream;
  HT_ASSERT(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;
  launch_params.params.residual = residual_.is_defined() ? residual_->raw_data_ptr() : nullptr;
  launch_params.params.rowscale = rowscale_.is_defined() ? rowscale_->raw_data_ptr() : nullptr;
  launch_params.params.colscale = colscale_.is_defined() ? colscale_->raw_data_ptr() : nullptr;
  launch_params.params.x0_subset = x0_subset_.is_defined() ? x0_subset_->raw_data_ptr() : nullptr;
  launch_params.params.z_subset = z_subset_.is_defined() ? z_subset_->raw_data_ptr() : nullptr;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int multiple = hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
  // Request the kernel launcher.
  auto launcher = get_fwd_launcher(wtype, itype, rtype, otype, ctype, round_multiple(hidden_size, multiple));

  // Set the kernel runtime parameters.
  layer_norm::FwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x0 = x0->raw_data_ptr();
  params.x = save_x ? x->raw_data_ptr() : nullptr;
  params.dmask = dropout_p > 0.f ? dmask->raw_data_ptr() : nullptr;
  params.mu = mu->raw_data_ptr();
  params.rs = rsigma->raw_data_ptr();
  params.gamma = gamma->raw_data_ptr();
  params.beta = beta_.is_defined() ? beta_->raw_data_ptr() : nullptr;
  params.z = z->raw_data_ptr();
  params.epsilon = epsilon;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.rowscale_const = rowscale_const;
  params.is_rms_norm = is_rms_norm;

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  NDArray workspace, barrier;

  if (dropout_p > 0.f) {
      // number of times random will be generated per thread, to offset philox counter in thc random
      // state
      int64_t counter_offset = launch_params.elts_per_thread;

      // See Note [Acquire lock when using random generators]
      {
        params.philox_args = std::pair<uint64_t, uint64_t>(
        hetu::impl::GenNextRandomSeed(), counter_offset);
      }
  }

  if( int64_t(launch_params.barrier_size) > 0 ) {
      HTShape barrier_shape{int64_t(launch_params.barrier_size)}, workspace_shape{int64_t(launch_params.workspace_bytes)};
      barrier = NDArray::zeros(barrier_shape, x0->device(), kInt32, stream.stream_index());
      workspace = NDArray::empty(workspace_shape, x0->device(), kUInt8, stream.stream_index());
      params.workspace = workspace->raw_data_ptr();
      params.barrier = barrier->data_ptr<int>();
  }

  // Launch the kernel.
  launcher(launch_params, false);

}

void DropoutAddLnBwdCuda(const NDArray& dz,     // BxSxhidden_size
                         const NDArray& dx_,     // BxSxhidden_size
                         const NDArray& x,      // BxSxhidden_size
                         const NDArray& x0_,     // BxSxhidden_size
                         const NDArray& dmask_,  // BxSxhidden_size
                         const NDArray& mu,     // BxS, FP32!
                         const NDArray& rsigma, // BxS, FP32!
                         const NDArray& gamma,   // hidden_size
                         const NDArray& rowscale_,      // BxS
                         const NDArray& colscale_,      // hidden_size
                         const NDArray& x0_subset_,      // BxS
                         const NDArray& z_subset_,      // BxS
                         NDArray& dx0, 
                         NDArray& dresidual, 
                         NDArray& dgamma, 
                         NDArray& dbeta, 
                         NDArray& dgamma_part, 
                         NDArray& dbeta_part,
                         NDArray& dcolscale,
                         NDArray& dcolscale_part,
                         const float dropout_p,
                         const float rowscale_const,
                         const int64_t x0_numrows,
                         const bool has_residual,
                         bool is_rms_norm,
                         const Stream& stream) {

    HT_LOG_TRACE << "dz = " << dz << ", dx_ = " << dx_  << ", x = " << x << ", x0_ = " << x0_ << ", dmask_ = " << dmask_ 
	<< ", gamma = " << gamma << ", mu = " << mu << ", rsigma = " << rsigma
	<< ", rowscale_ = " << rowscale_ << ", colscale_ = " << colscale_ << ", dx0 = " << dx0
	<< ", dresidual = " << dresidual << ", dgamma = " << dgamma << ", dbeta = " << dbeta;
    auto itype = dz->dtype();
    auto rtype = x->dtype();
    auto wtype = gamma->dtype();
    auto otype = itype;
    auto ctype = kFloat32;
    auto mtype = kUInt8;

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    if (dropout_p > 0.f) { HT_ASSERT(dmask_.is_defined()); }

    HT_ASSERT(dz->dtype() == otype);
    HT_ASSERT(mu->dtype() == ctype);
    HT_ASSERT(rsigma->dtype() == ctype);

    HT_ASSERT(x->is_cuda());
    HT_ASSERT(dz->is_cuda());
    HT_ASSERT(mu->is_cuda());
    HT_ASSERT(rsigma->is_cuda());
    HT_ASSERT(gamma->is_cuda());

    HT_ASSERT(x->is_contiguous());
    HT_ASSERT(dz->is_contiguous());

    auto sizes = x->shape();
    HT_ASSERT(sizes.size() == 2);
    auto rows = sizes[0];
    auto cols = sizes[1];
    HT_ASSERT(dz->ndim() == 2);
    HT_ASSERT(dz->shape(1) == cols);
    auto hidden_size = gamma->numel();
    HT_ASSERT(hidden_size == cols);

    HT_LOG_TRACE << "rms 1";
    //  does not own the storage, so we need to construct a vector.
    // Otherwise just constructing IntArrayRef({blah}) will cause uninitialized memory because
    // blah is then deallocated.
    HTShape x0_sizes {!x0_subset_.is_defined() ? rows : x0_numrows, cols};
    HTShape rows_shape{rows};
    HTShape cols_shape{cols};

    if (dx_.is_defined()) {
        auto dx = dx_;
        HT_ASSERT(dx->dtype() == rtype);
        HT_ASSERT(dx->is_cuda());
        HT_ASSERT(dx->is_contiguous());
        HT_ASSERT(dx->shape() == sizes);
    }

    if (dmask_.is_defined()) {
        auto dmask = dmask_;
        HT_ASSERT(dmask->dtype() == mtype);
        HT_ASSERT(dmask->is_cuda());
        HT_ASSERT(dmask->is_contiguous());
        HT_ASSERT(dmask->shape() == x0_sizes);
    }

    if (rowscale_.is_defined()) {
        auto rowscale = rowscale_;
        HT_ASSERT(rowscale->is_cuda());
        HT_ASSERT(rowscale->is_contiguous());
        HT_ASSERT(rowscale->shape() == rows_shape);
        HT_ASSERT(rowscale->dtype() == itype);
    }

    if (colscale_.is_defined()) {
        auto colscale = colscale_;
        HT_ASSERT(colscale->is_cuda());
        HT_ASSERT(colscale->is_contiguous());
        HT_ASSERT(colscale->shape() == cols_shape);
        HT_ASSERT(colscale->dtype() == wtype);

        HT_ASSERT(x0_.is_defined());
        auto x0 = x0_;
        HT_ASSERT(x0->is_cuda());
        HT_ASSERT(x0->is_contiguous());
        HT_ASSERT(x0->shape() == x0_sizes);
        HT_ASSERT(x0->dtype() == itype);
    }

    if (x0_subset_.is_defined()) {
        auto x0_subset = x0_subset_;
        HT_ASSERT(x0_subset->is_cuda());
        HT_ASSERT(x0_subset->is_contiguous());
        HT_ASSERT(x0_subset->shape() == rows_shape);
        HT_ASSERT(x0_subset->dtype() == kInt32);

        HT_ASSERT(z_subset_.is_defined());
        auto z_subset = z_subset_;
        HT_ASSERT(z_subset->is_cuda());
        HT_ASSERT(z_subset->is_contiguous());
        HT_ASSERT(z_subset->shape() == rows_shape);
        HT_ASSERT(z_subset->dtype() == kInt32);
    }
    HT_LOG_TRACE << "rms 2";
    HT_ASSERT((hidden_size % 8 == 0) && (hidden_size <= 8192));

    HT_ASSERT(mu->numel() == rows);
    HT_ASSERT(mu->shape() == rsigma->shape());

    HT_ASSERT(gamma->numel() == cols);

    // auto dx0 = NDArray::empty(x0_sizes, x->device(), itype, stream.stream_index());
    // NDArray dresidual;
    // if (has_residual) { dresidual = NDArray::empty(x->shape(), x->device(), rtype, stream.stream_index()); }
    // auto dgamma = NDArray::empty_like(gamma, stream.stream_index());
    // auto dbeta = NDArray::empty_like(gamma, stream.stream_index());
    // NDArray dcolscale;
    // if (colscale_.is_defined()) {
    //     dcolscale = NDArray::empty_like(colscale_, stream.stream_index());
    // }

    HT_LOG_TRACE << "rms 3";
    layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
    launch_params.stream = cuda_stream;
    cudaDeviceProp prop = Device::dprop(x->device().index());
    launch_params.props = &prop;
    HT_ASSERT(dropout_p < 1.f);
    launch_params.params.dropout_keep_p = 1.f - dropout_p;
    launch_params.params.dresidual = has_residual ? dresidual->raw_data_ptr() : nullptr;
    launch_params.params.rowscale = rowscale_.is_defined() ? rowscale_->raw_data_ptr() : nullptr;
    launch_params.params.colscale = colscale_.is_defined() ? colscale_->raw_data_ptr() : nullptr;
    launch_params.params.x0_subset = x0_subset_.is_defined() ? x0_subset_->raw_data_ptr() : nullptr;
    launch_params.params.z_subset = z_subset_.is_defined() ? z_subset_->raw_data_ptr() : nullptr;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int multiple = hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
    auto launcher = get_bwd_launcher(wtype, itype, rtype, otype, ctype, round_multiple(hidden_size, multiple));

    HT_LOG_TRACE << "rms 4";
    launcher(launch_params, true);
    HTShape part_shape = {int64_t(launch_params.params.ctas_per_col), int64_t(hidden_size)};
    dgamma_part = NDArray::empty(part_shape, x->device(), ctype, stream.stream_index());
    dbeta_part = NDArray::empty(part_shape, x->device(), ctype, stream.stream_index());
    if (colscale_.is_defined()) {
        dcolscale_part = NDArray::empty(part_shape, x->device(), ctype, stream.stream_index());
    }
    NDArray workspace, barrier;

    layer_norm::BwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = x->raw_data_ptr();
    params.x0 = x0_.is_defined() ? x0_->raw_data_ptr() : nullptr;
    params.dmask = dropout_p > 0.f ? dmask_->raw_data_ptr() : nullptr;
    params.mu = mu->raw_data_ptr();
    params.rs = rsigma->raw_data_ptr();
    params.gamma = gamma->raw_data_ptr();
    params.dz = dz->raw_data_ptr();
    params.dx = dx_.is_defined() ? dx_->raw_data_ptr() : nullptr;
    params.dx0 = dx0->raw_data_ptr();
    params.dbeta = dbeta->raw_data_ptr();
    params.dgamma = dgamma->raw_data_ptr();
    params.dcolscale = colscale_.is_defined() ? dcolscale->raw_data_ptr() : nullptr;
    params.dbeta_part = dbeta_part->raw_data_ptr();
    params.dgamma_part = dgamma_part->raw_data_ptr();
    params.dcolscale_part = colscale_.is_defined() ? dcolscale_part->raw_data_ptr() : nullptr;
    params.dropout_scale = 1.f / (1.f - dropout_p);
    params.inverse_cols = 1.f / float(params.cols);
    params.rowscale_const = rowscale_const;
    params.is_rms_norm = is_rms_norm;
    HT_LOG_TRACE << "rms 5";
    if( int64_t(launch_params.barrier_size) > 0 ) {
        // TODO Any way to avoid this?
        HTShape barrier_shape{int64_t(launch_params.barrier_size)}, workspace_shape{int64_t(launch_params.workspace_bytes)};
        barrier = NDArray::zeros(barrier_shape, x->device(), kInt32, stream.stream_index());
        workspace = NDArray::empty(workspace_shape, x->device(), kUInt8, stream.stream_index());
        params.workspace = workspace->raw_data_ptr();
        params.barrier = barrier->data_ptr<int>();
    }

    launcher(launch_params, false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DropoutAddLnParallelResidualFwdCuda(
  const NDArray& x0,      // Input: BxSxhidden_size
  const NDArray& x1_,      // Input: BxSxhidden_size
  const NDArray& residual_,  // Residual: BxSxhidden_size
  const NDArray& gamma0,   // hidden_size
  const NDArray& beta0_,   // hidden_size
  const NDArray& gamma1_,   // hidden_size
  const NDArray& beta1_,   // hidden_size
  NDArray& z0,
  NDArray& z1,
  NDArray& x,
  NDArray& dmask0,
  NDArray& dmask1,
  NDArray& mu,
  NDArray& rsigma,
  const float dropout_p,
  const float epsilon,
  bool residual_in_fp32,
  bool is_rms_norm,
  const Stream& stream) {
    auto itype = x0->dtype();
    auto rtype = residual_.is_defined()
        ? residual_->dtype()
        : (residual_in_fp32 ? kFloat32 : x0->dtype());
    auto wtype = gamma0->dtype();
    auto otype = itype;
    auto ctype = kFloat32;
    auto mtype = kUInt8;

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    HT_ASSERT(x0->is_cuda());
    HT_ASSERT(gamma0->is_cuda());

    HT_ASSERT(x0->is_contiguous());
    const auto sizes = x0->shape();
    HT_ASSERT(x0->ndim() == 2);

    const int rows = sizes[0];
    const int cols = sizes[1];
    auto hidden_size = gamma0->numel();
    HT_ASSERT(hidden_size == cols);
    HTShape rows_shape{rows};
    HTShape cols_shape{cols};

    if (x1_.is_defined()) {
        auto x1 = x1_;
        HT_ASSERT(x1->is_cuda());
        HT_ASSERT(x1->is_contiguous());
        HT_ASSERT(x1->shape() == sizes);
    }

    if (residual_.is_defined()) {
        auto residual = residual_;
        HT_ASSERT(residual->is_cuda());
        HT_ASSERT(residual->is_contiguous());
        HT_ASSERT(residual->shape() == sizes);
    }

    if (beta0_.is_defined()) {
        auto beta0 = beta0_;
        HT_ASSERT(beta0->dtype() == wtype);
        HT_ASSERT(beta0->is_cuda());
        HT_ASSERT(beta0->is_contiguous());
        HT_ASSERT(beta0->shape() == gamma0->shape());
    }

    if (gamma1_.is_defined()) {
        auto gamma1 = gamma1_;
        HT_ASSERT(gamma1->dtype() == wtype);
        HT_ASSERT(gamma1->is_cuda());
        HT_ASSERT(gamma1->is_contiguous());
        HT_ASSERT(gamma1->shape() == gamma0->shape());
    }

    if (beta1_.is_defined()) {
        auto beta1 = beta1_;
        HT_ASSERT(beta1->dtype() == wtype);
        HT_ASSERT(beta1->is_cuda());
        HT_ASSERT(beta1->is_contiguous());
        HT_ASSERT(beta1->shape() == gamma0->shape());
    }

    HT_ASSERT((hidden_size % 8 == 0) && (hidden_size <= 8192));
    HT_ASSERT(epsilon >= 0.f);

    bool save_x = residual_.is_defined() || x1_.is_defined() || (dropout_p > 0.f) || (itype != rtype);
    // NDArray x;
    // if (save_x) { x = NDArray::empty(sizes, x0->device(), rtype, stream.stream_index()); }
    // NDArray dmask0, dmask1;
    // if (dropout_p > 0.f) {
    //     dmask0 = NDArray::empty(x0->shape(), x0->device(), otype, stream.stream_index());
    //     if (x1_.is_defined()) { dmask1 = NDArray::empty(x0->shape(), x0->device(), mtype, stream.stream_index()); }
    // };
    // auto z0 = NDArray::empty(sizes, x0->device(), otype, stream.stream_index());
    // NDArray z1;
    // if (gamma1_.is_defined()) { z1 = NDArray::empty(sizes, x0->device(), otype, stream.stream_index()); }

    // auto mu = NDArray::empty({ rows }, x0->device(), ctype, stream.stream_index());
    // auto rsigma = NDArray::empty({ rows }, x0->device(), ctype, stream.stream_index());

    layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

    cudaDeviceProp prop = Device::dprop(x0->device().index());
    launch_params.props = &prop;
    launch_params.stream = cuda_stream;
    HT_ASSERT(dropout_p < 1.f);
    launch_params.params.dropout_keep_p = 1.f - dropout_p;
    launch_params.params.residual = residual_.is_defined() ? residual_->raw_data_ptr() : nullptr;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int multiple = hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
    // Request the kernel launcher.
    auto launcher = get_parallel_fwd_launcher(wtype, itype, rtype, otype, ctype, round_multiple(hidden_size, multiple));

    // Set the kernel runtime parameters.
    layer_norm::FwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x0 = x0->raw_data_ptr();
    params.x1 = x1_.is_defined() ? x1_->raw_data_ptr() : nullptr;
    params.x = save_x ? x->raw_data_ptr() : nullptr;
    params.dmask = dropout_p > 0.f ? dmask0->raw_data_ptr() : nullptr;
    params.dmask1 = (dropout_p > 0.f && x1_.is_defined()) ? dmask1->raw_data_ptr() : nullptr;
    params.mu = mu->raw_data_ptr();
    params.rs = rsigma->raw_data_ptr();
    params.gamma = gamma0->raw_data_ptr();
    params.gamma1 = gamma1_.is_defined() ? gamma1_->raw_data_ptr() : nullptr;
    params.beta = beta0_.is_defined() ? beta0_->raw_data_ptr() : nullptr;
    params.beta1 = beta1_.is_defined() ? beta1_->raw_data_ptr() : nullptr;
    params.z = z0->raw_data_ptr();
    params.z1 = gamma1_.is_defined() ? z1->raw_data_ptr() : nullptr;
    params.epsilon = epsilon;
    params.dropout_scale = 1.f / (1.f - dropout_p);
    params.inverse_cols = 1.f / float(params.cols);
    params.is_rms_norm = is_rms_norm;

    // Query the kernel-specific launch parameters.
    launcher(launch_params, true);

    NDArray workspace, barrier;

    if (dropout_p > 0.f) {
        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        int64_t counter_offset = 2 * launch_params.elts_per_thread;

        // See Note [Acquire lock when using random generators]
        {
          params.philox_args = std::pair<uint64_t, uint64_t>(
          hetu::impl::GenNextRandomSeed(), counter_offset);
        }
    }

    if( int64_t(launch_params.barrier_size) > 0 ) {
        HTShape barrier_shape{int64_t(launch_params.barrier_size)}, workspace_shape{int64_t(launch_params.workspace_bytes)};
        barrier = NDArray::zeros(barrier_shape, x0->device(), kInt32, stream.stream_index());
        workspace = NDArray::empty(workspace_shape, x0->device(), kUInt8, stream.stream_index());
        params.workspace = workspace->raw_data_ptr();
        params.barrier = barrier->data_ptr<int>();
    }

    // Launch the kernel.
    launcher(launch_params, false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<NDArray> DropoutAddLnParallelResidualBwdCuda(
    const NDArray& dz0,     // BxSxhidden_size
    const NDArray& dz1_,     // BxSxhidden_size
    const NDArray& dx_,     // BxSxhidden_size
    const NDArray& x,      // BxSxhidden_size
    const NDArray& dmask0_,  // BxSxhidden_size
    const NDArray& dmask1_,  // BxSxhidden_size
    const NDArray& mu,     // BxS, FP32!
    const NDArray& rsigma, // BxS, FP32!
    const NDArray& gamma0,   // hidden_size
    const NDArray& gamma1_,   // hidden_size
    NDArray& dx0,
    NDArray& dx1,
    NDArray& dresidual,
    NDArray& dgamma0,
    NDArray& dbeta0,
    NDArray& dgamma1,
    NDArray& dbeta1,
    NDArray& dgamma0_part,
    NDArray& dbeta0_part,
    NDArray& dgamma1_part,
    NDArray& dbeta1_part, 
    const float dropout_p,
    const bool has_x1,
    const bool has_residual,
    bool is_rms_norm,
    const Stream& stream) {

    auto itype = dz0->dtype();
    auto rtype = x->dtype();
    auto wtype = gamma0->dtype();
    auto otype = itype;
    auto ctype = kFloat32;
    auto mtype = kUInt8;

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

    if (dropout_p > 0.f) { HT_ASSERT(dmask0_.is_defined()); }

    HT_ASSERT(dz0->dtype() == otype);
    HT_ASSERT(dz0->dtype() == otype);
    HT_ASSERT(mu->dtype() == ctype);
    HT_ASSERT(rsigma->dtype() == ctype);

    HT_ASSERT(x->is_cuda());
    HT_ASSERT(dz0->is_cuda());
    HT_ASSERT(mu->is_cuda());
    HT_ASSERT(rsigma->is_cuda());
    HT_ASSERT(gamma0->is_cuda());

    HT_ASSERT(x->is_contiguous());
    HT_ASSERT(dz0->is_contiguous());

    auto sizes = x->shape();
    HT_ASSERT(sizes.size() == 2);
    auto rows = sizes[0];
    auto cols = sizes[1];
    HT_ASSERT(dz0->ndim() == 2);
    HT_ASSERT(dz0->shape(1) == cols);
    auto hidden_size = gamma0->numel();
    HT_ASSERT(hidden_size == cols);
    HTShape rows_shape{rows};
    HTShape cols_shape{cols};

    if (dz1_.is_defined()) {
        auto dz1 = dz1_;
        HT_ASSERT(dz1->dtype() == otype);
        HT_ASSERT(dz1->is_cuda());
        HT_ASSERT(dz1->is_contiguous());
        HT_ASSERT(dz1->shape() == sizes);

        HT_ASSERT(gamma1_.is_defined());
        auto gamma1 = gamma1_;
        HT_ASSERT(gamma1->dtype() == wtype);
        HT_ASSERT(gamma1->is_cuda());
        HT_ASSERT(gamma1->is_contiguous());
        HT_ASSERT(gamma1->shape() == gamma0->shape());
    }

    if (dx_.is_defined()) {
        auto dx = dx_;
        HT_ASSERT(dx->dtype() == rtype);
        HT_ASSERT(dx->is_cuda());
        HT_ASSERT(dx->is_contiguous());
        HT_ASSERT(dx->shape() == sizes);
    }

    if (dmask0_.is_defined()) {
        auto dmask0 = dmask0_;
        HT_ASSERT(dmask0->dtype() == mtype);
        HT_ASSERT(dmask0->is_cuda());
        HT_ASSERT(dmask0->is_contiguous());
        HT_ASSERT(dmask0->shape() == sizes);

        if (has_x1) {
            HT_ASSERT(dmask1_.is_defined());
            auto dmask1 = dmask1_;
            HT_ASSERT(dmask1->dtype() == mtype);
            HT_ASSERT(dmask1->is_cuda());
            HT_ASSERT(dmask1->is_contiguous());
            HT_ASSERT(dmask1->shape() == sizes);
        }
    }

    HT_ASSERT((hidden_size % 8 == 0) && (hidden_size <= 8192));

    HT_ASSERT(mu->numel() == rows);
    HT_ASSERT(mu->shape() == rsigma->shape());

    // auto dx0 = NDArray::empty(sizes, x0->device(), itype, stream.stream_index());
    // NDArray dx1;
    // if (has_x1) { dx1 = NDArray::empty(sizes, x0->device(), itype, stream.stream_index()); }
    // NDArray dresidual;
    // if (has_residual) { dresidual = NDArray::empty_like(x, x0->device(), rtype, stream.stream_index()); }
    // auto dgamma0 = NDArray::empty_like(gamma0, stream.stream_index());
    // auto dbeta0 = NDArray::empty_like(gamma0, stream.stream_index());
    // NDArray dgamma1, dbeta1;
    // if (gamma1_.is_defined()) {
    //     dgamma1 = NDArray::empty_like(gamma0, stream.stream_index());
    //     dbeta1 = NDArray::empty_like(gamma0, stream.stream_index());
    // }

    layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
    launch_params.stream = cuda_stream;
    cudaDeviceProp prop = Device::dprop(x->device().index());
    launch_params.props = &prop;
    HT_ASSERT(dropout_p < 1.f);
    launch_params.params.dropout_keep_p = 1.f - dropout_p;
    launch_params.params.dresidual = has_residual ? dresidual->raw_data_ptr() : nullptr;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int multiple = hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
    auto launcher = get_parallel_bwd_launcher(wtype, itype, rtype, otype, ctype, round_multiple(hidden_size, multiple));

    launcher(launch_params, true);

    // auto dgamma0_part = NDArray::zeros({ launch_params.params.ctas_per_col, hidden_size }, x0->device(), ctype, stream.stream_index());
    // auto dbeta0_part = NDArray::zeros({ launch_params.params.ctas_per_col, hidden_size }, x0->device(), ctype, stream.stream_index());
    // NDArray dgamma1_part, dbeta1_part;
    // if (gamma1_.is_defined()) {
    //     dgamma1_part = NDArray::zeros_like(dgamma0_part, stream.stream_index());
    //     dbeta1_part = NDArray::zeros_like(dbeta0_part, stream.stream_index());
    // }
    NDArray workspace, barrier;

    layer_norm::BwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = x->raw_data_ptr();
    params.dmask = dropout_p > 0.f ? dmask0_->raw_data_ptr() : nullptr;
    params.dmask1 = (dropout_p > 0.f && has_x1) ? dmask1_->raw_data_ptr() : nullptr;
    params.mu = mu->raw_data_ptr();
    params.rs = rsigma->raw_data_ptr();
    params.gamma = gamma0->raw_data_ptr();
    params.gamma1 = gamma1_.is_defined() ? gamma1_->raw_data_ptr() : nullptr;
    params.dz = dz0->raw_data_ptr();
    params.dz1 = dz1_.is_defined() ? dz1_->raw_data_ptr() : nullptr;
    params.dx = dx_.is_defined() ? dx_->raw_data_ptr() : nullptr;
    params.dx0 = dx0->raw_data_ptr();
    params.dx1 = has_x1 ? dx1->raw_data_ptr() : nullptr;
    params.dbeta = dbeta0->raw_data_ptr();
    params.dgamma = dgamma0->raw_data_ptr();
    params.dbeta1 = gamma1_.is_defined() ? dbeta1->raw_data_ptr() : nullptr;
    params.dgamma1 = gamma1_.is_defined() ? dgamma1->raw_data_ptr() : nullptr;
    params.dbeta_part = dbeta0_part->raw_data_ptr();
    params.dgamma_part = dgamma0_part->raw_data_ptr();
    params.dbeta1_part = gamma1_.is_defined() ? dbeta1_part->raw_data_ptr() : nullptr;
    params.dgamma1_part = gamma1_.is_defined() ? dgamma1_part->raw_data_ptr() : nullptr;
    params.dropout_scale = 1.f / (1.f - dropout_p);
    params.inverse_cols = 1.f / float(params.cols);
    params.is_rms_norm = is_rms_norm;

    if( int64_t(launch_params.barrier_size) > 0 ) {
        // TODO Any way to avoid this?
        HTShape barrier_shape{int64_t(launch_params.barrier_size)}, workspace_shape{int64_t(launch_params.workspace_bytes)};
        barrier = NDArray::zeros(barrier_shape, x->device(), kInt32, stream.stream_index());
        workspace = NDArray::empty(workspace_shape, x->device(), kUInt8, stream.stream_index());
        params.workspace = workspace->raw_data_ptr();
        params.barrier = barrier->data_ptr<int>();
    }

    launcher(launch_params, false);
}

} // namespace impl
} // namespace hetu

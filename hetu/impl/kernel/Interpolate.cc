#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
spec_t cubic_convolution1(const spec_t x, const spec_t a) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

template <typename spec_t>
spec_t cubic_convolution2(const spec_t x, const spec_t a) {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

template <typename spec_t>
spec_t kecubic_interp(spec_t x0, spec_t x1, spec_t x2, spec_t x3,
                      spec_t t) {
    spec_t coeffs[4];
    spec_t A = -0.75;
    spec_t y1 = t;
    spec_t y2 = 1.0 - t;
    coeffs[0] = cubic_convolution2(spec_t(y1 + 1.0), A);
    coeffs[1] = cubic_convolution1(y1, A);
    coeffs[2] = cubic_convolution1(y2, A);
    coeffs[3] = cubic_convolution2(spec_t(y2 + 1.0), A);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename spec_t>
void get_cubic_upsample_coefficients(spec_t coeffs[4], spec_t t) {
    spec_t A = -0.75;
    spec_t y1 = t;
    spec_t y2 = 1.0 - t;
    coeffs[0] = cubic_convolution2(spec_t(y1 + 1.0), A);
    coeffs[1] = cubic_convolution1(y1, A);
    coeffs[2] = cubic_convolution1(y2, A);
    coeffs[3] = cubic_convolution2(spec_t(y2 + 1.0), A);
}


template <typename spec_t>
void interpolate_cpu(const spec_t *input, int64_t n, int64_t c,
                     int64_t in_h, int64_t in_w, spec_t *output,
                     int64_t out_h, int64_t out_w, spec_t ratio_h,
                     spec_t ratio_w, bool align_corners, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {

    int64_t in_hw = in_h * in_w;
    int64_t out_hw = out_h * out_w;
    int64_t in_chw = c * in_hw;
    int64_t out_chw = c * out_hw;

    int64_t n_id = idx / out_chw;
    int64_t chw_id = idx % out_chw;
    int64_t c_id = chw_id / out_hw;
    int64_t hw_id = chw_id % out_hw;
    int64_t out_h_id = hw_id / out_w;
    int64_t out_w_id = hw_id % out_w;

    spec_t in_h_id_ = align_corners ? spec_t(ratio_h * out_h_id) :
                                      spec_t(ratio_h * (out_h_id + 0.5) - 0.5);
    int64_t in_h_id = std::floor(in_h_id_);
    spec_t in_h_delta = in_h_id_ - in_h_id;

    spec_t in_w_id_ = align_corners ? spec_t(ratio_w * out_w_id) :
                                      spec_t(ratio_w * (out_w_id + 0.5) - 0.5);
    int64_t in_w_id = std::floor(in_w_id_);
    spec_t in_w_delta = in_w_id_ - in_w_id;

    spec_t coefficients[4];
    spec_t x0, x1, x2, x3;

    for (int64_t k = 0; k < 4; k++) {
        int64_t tmp_h = std::max(std::min(in_h_id - 1 + k, in_h - 1), int64_t(0));
        int64_t tmp_w0 = std::max(std::min(in_w_id - 1, in_w - 1), int64_t(0));
        int64_t tmp_w1 = std::max(std::min(in_w_id + 0, in_w - 1), int64_t(0));
        int64_t tmp_w2 = std::max(std::min(in_w_id + 1, in_w - 1), int64_t(0));
        int64_t tmp_w3 = std::max(std::min(in_w_id + 2, in_w - 1), int64_t(0));

        x0 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w0];
        x1 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w1];
        x2 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w2];
        x3 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w3];

        coefficients[k] = kecubic_interp(x0, x1, x2, x3, in_w_delta);
    }
    spec_t val =
        kecubic_interp(coefficients[0], coefficients[1], coefficients[2],
                        coefficients[3], in_h_delta);
    output[idx] = val;
  }
}

template <typename spec_t>
void array_zero_set_cpu(spec_t* input, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) 
    input[idx] = 0;
}

template <typename spec_t>
void interpolate_gradient_cpu(const spec_t *output, int64_t n, int64_t c,
                              int64_t in_h, int64_t in_w, spec_t *input,
                              int64_t out_h, int64_t out_w, spec_t ratio_h,
                              spec_t ratio_w, bool align_corners, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) {

    int64_t in_hw = in_h * in_w;
    int64_t out_hw = out_h * out_w;
    int64_t in_chw = c * in_hw;
    int64_t out_chw = c * out_hw;

    int64_t n_id = idx / out_chw;
    int64_t chw_id = idx % out_chw;
    int64_t c_id = chw_id / out_hw;
    int64_t hw_id = chw_id % out_hw;
    int64_t out_h_id = hw_id / out_w;
    int64_t out_w_id = hw_id % out_w;

    spec_t in_h_id_ = align_corners ? spec_t(ratio_h * out_h_id) :
                                      spec_t(ratio_h * (out_h_id + 0.5) - 0.5);
    int64_t in_h_id = std::floor(in_h_id_);
    spec_t in_h_delta = in_h_id_ - in_h_id;

    spec_t in_w_id_ = align_corners ? spec_t(ratio_w * out_w_id) :
                                      spec_t(ratio_w * (out_w_id + 0.5) - 0.5);
    int64_t in_w_id = std::floor(in_w_id_);
    spec_t in_w_delta = in_w_id_ - in_w_id;

    spec_t coeffs_h[4];
    spec_t coeffs_w[4];

    get_cubic_upsample_coefficients(coeffs_h, in_h_delta);
    get_cubic_upsample_coefficients(coeffs_w, in_w_delta);

    for (int64_t i = 0; i < 4; i++) {
      for (int64_t j = 0; j < 4; j++) {
        int64_t tmp_h = std::max(std::min(in_h_id - 1 + i, in_h - 1), int64_t(0));
        int64_t tmp_w = std::max(std::min(in_w_id - 1 + j, in_w - 1), int64_t(0));
        spec_t addend = output[idx] * coeffs_h[i] * coeffs_w[j];
        input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w] += addend;
      }
    }
  }
}

void InterpolateCpu(const NDArray& input, NDArray& output,
                    bool align_corners, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  int64_t input_N = input->shape(0);
  int64_t input_C = input->shape(1);
  int64_t input_H = input->shape(2);
  int64_t input_W = input->shape(3);
  int64_t output_H = output->shape(2);
  int64_t output_W = output->shape(3);

  float ratio_h = 0.f;
  float ratio_w = 0.f;

  ratio_h = (align_corners) ? (float)(input_H - 1) / (output_H - 1) :
                              (float)(input_H) / output_H;

  ratio_w = (align_corners) ? (float)(input_W - 1) / (output_W - 1) :
                              (float)(input_W) / output_W;

  int64_t size = input_N * input_C * output_H * output_W;
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "InterpolateCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, input_N, input_C, input_H, input_W,
      output_H, output_W, ratio_h, ratio_w, align_corners, size]() {
      interpolate_cpu<spec_t>(
        input->data_ptr<spec_t>(), input_N, input_C, input_H, input_W, output->data_ptr<spec_t>(),
        output_H, output_W, static_cast<spec_t>(ratio_h), static_cast<spec_t>(ratio_w), align_corners, size);
      },"Interpolate");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void InterpolateGradientCpu(const NDArray& output, NDArray& input,
                            bool align_corners, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  
  int64_t input_N = input->shape(0);
  int64_t input_C = input->shape(1);
  int64_t input_H = input->shape(2);
  int64_t input_W = input->shape(3);
  int64_t output_H = output->shape(2);
  int64_t output_W = output->shape(3);

  float ratio_h = 0.f;
  float ratio_w = 0.f;

  ratio_h = (align_corners) ? (float)(input_H - 1) / (output_H - 1) :
                              (float)(input_H) / output_H;

  ratio_w = (align_corners) ? (float)(input_W - 1) / (output_W - 1) :
                              (float)(input_W) / output_W;

  int64_t size = input_N * input_C * output_H * output_W;
  if (size == 0)
    return;

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "InterpolateGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, input_N, input_C, input_H, input_W,
      output_H, output_W, ratio_h, ratio_w, align_corners, size]() {
      array_zero_set_cpu<spec_t>(
        input->data_ptr<spec_t>(), input->numel());
      interpolate_gradient_cpu<spec_t>(
        output->data_ptr<spec_t>(), input_N, input_C, input_H, input_W, input->data_ptr<spec_t>(),
        output_H, output_W, static_cast<spec_t>(ratio_h), static_cast<spec_t>(ratio_w), align_corners, size);
      },"Interpolate");
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu

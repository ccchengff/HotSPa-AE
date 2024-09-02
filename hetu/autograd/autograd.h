#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

TensorList Gradients(const TensorList& ys, const TensorList& xs,
                     const TensorList& grad_ys = {});

inline TensorList Gradients(const Tensor& y, const TensorList& xs,
                            const Tensor& grad_y = Tensor()) {
  return Gradients(TensorList({y}), xs, TensorList({grad_y}));
}

TensorList _FillGrads(const TensorList& edges, const TensorList& grads);

TensorList _Filter(const TensorList& nodes);

Tensor _Sum(const TensorList& nodes);

} // namespace autograd
} // namespace hetu

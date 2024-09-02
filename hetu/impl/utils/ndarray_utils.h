#pragma once

#include "hetu/core/ndarray.h"

#define HT_ASSERT_CUDA_DEVICE(x)                                               \
  HT_ASSERT(x->is_cuda()) << "Data " << #x << " is not on a CUDA device."
#define HT_ASSERT_CPU_DEVICE(x)                                                \
  HT_ASSERT(x->is_cpu()) << "Data " << #x << " is not on a host device."
#define HT_ASSERT_SAME_SHAPE(x, y)                                             \
  HT_ASSERT(x->shape() == y->shape())                                          \
    << "Data are not with the same shape: "                                    \
    << "(" << #x << ") " << x->shape() << " vs. (" << #y << ") " << y->shape() \
    << "."
#define HT_ASSERT_SAME_DTYPE(x, y)                                             \
  HT_ASSERT(x->dtype() == y->dtype())                                          \
    << "Data are not with the same data type: "                                \
    << "(" << #x << ") " << x->dtype() << " vs. (" << #y << ") " << y->dtype() \
    << "."
#define HT_ASSERT_SAME_DEVICE(x, y)                                            \
  HT_ASSERT(x->device() == y->device())                                        \
    << "Data are not on the same device: "                                     \
    << "(" << #x << ") " << x->device() << " vs. (" << #y << ") "              \
    << y->device() << "."
#define HT_ASSERT_NDIM(x, nd)                                                  \
  HT_ASSERT(x->ndim() == (nd)) << "Data << " #x << " expected to be with "     \
                               << (nd) << " dimensions but got " << x->ndim();
#define HT_ASSERT_SAME_NDIM(x, y)                                              \
  HT_ASSERT(x->ndim() == y->ndim())                                            \
    << "Data are not with the same number of dimensions: "                     \
    << "(" << #x << ") " << x->ndim() << " vs. (" << #y << ") " << y->ndim()   \
    << "."
#define HT_ASSERT_COPIABLE(x, y)                                               \
  HT_ASSERT(IsCopiable(x, y)) << "Data are not copiable: "                     \
                              << "(" << #x << ") " << x->meta() << " vs. ("    \
                              << #y << ") " << y->meta() << "."
#define HT_ASSERT_EXCHANGABLE(x, y)                                            \
  HT_ASSERT(IsExchangable(x, y)) << "Data are not exchangable: "               \
                                 << "(" << #x << ") " << x->meta() << " vs. (" \
                                 << #y << ") " << y->meta() << "."
#define HT_ASSERT_CONTIGUOUS(x)                                                \
  HT_ASSERT(x->is_contiguous())  << "Data " << #x << " is not contiguous."     \
                                 << " Shape: " << x->shape() << "."            \
                                 << " Strides: " << x->stride() << "."
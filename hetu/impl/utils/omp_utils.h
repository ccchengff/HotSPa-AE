#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/device.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hetu {
namespace omp {

inline static void OMP_SET_NUM_THREADS(int num_thr) {
#ifdef _OPENMP
  omp_set_num_threads(num_thr);
#endif
}

inline static int OMP_GET_NUM_THREADS() {
  int ret = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  ret = omp_get_num_threads();
#endif
  return ret;
}

inline static int OMP_GET_THREAD_ID() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

} // namespace omp
} // namespace hetu

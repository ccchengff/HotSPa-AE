#pragma once

#include <chrono>
#include <ctime>
#include "hetu/common/logging.h"

/******************************************************
 * Timing Utils
 ******************************************************/

#define COST_NANOSEC(_X)                                                       \
  std::chrono::duration_cast<std::chrono::nanoseconds>((_X##_stop) -           \
                                                       (_X##_start))           \
    .count()
#define COST_MICROSEC(_X)                                                      \
  std::chrono::duration_cast<std::chrono::microseconds>((_X##_stop) -          \
                                                        (_X##_start))          \
    .count()
#define COST_MSEC(_X)                                                          \
  std::chrono::duration_cast<std::chrono::milliseconds>((_X##_stop) -          \
                                                        (_X##_start))          \
    .count()
#define COST_SEC(_X)                                                           \
  std::chrono::duration_cast<std::chrono::seconds>((_X##_stop) - (_X##_start)) \
    .count()
#define TIK(_X)                                                                \
  auto _X##_start = std::chrono::steady_clock::now(), _X##_stop = _X##_start
#define TOK(_X) _X##_stop = std::chrono::steady_clock::now()

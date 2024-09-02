#pragma once

#ifndef __cplusplus
#error "Cannot detect C++ version"
#endif

#if (__cplusplus >= 201703L)
#include <optional>
#elif (__cplusplus >= 201402L)
#include <experimental/optional>
#else
#error "Require C++14 or newer"
#endif

namespace hetu {

#if (__cplusplus >= 201703L)
template <typename T>
using optional = std::optional<T>;
using nullopt_t = std::nullopt_t;
constexpr nullopt_t nullopt = std::nullopt;
#elif (__cplusplus >= 201402L)
template <typename T>
using optional = std::experimental::optional<T>;
using nullopt_t = std::experimental::nullopt_t;
constexpr nullopt_t nullopt = std::experimental::nullopt;
#else
#error "Require C++14 or newer"
#endif

} // namespace hetu

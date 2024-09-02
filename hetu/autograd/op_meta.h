#pragma once

#include "hetu/core/device.h"
#include "hetu/core/stream.h"
#include "hetu/autograd/common.h"

namespace hetu {
namespace autograd {

class OpMeta {
 public:
  OpMeta() = default;
  OpMeta(const OpName& name, StreamIndex stream_index,
         const DeviceGroup& device_group, const TensorList& extra_deps) {
    set_name(name);
    set_stream_index(stream_index);
    set_device_group(device_group);
    set_extra_deps(extra_deps);
  }
  OpMeta(const OpMeta&) = default;
  OpMeta(OpMeta&&) = default;
  OpMeta& operator=(OpMeta&&) = default;
  OpMeta& operator=(const OpMeta&) = default;

  inline OpMeta& set_name(const OpName& n) {
    name = n;
    return *this;
  }

  inline OpMeta& set_name(OpName&& n) {
    name = std::move(n);
    return *this;
  }

  inline OpMeta& set_stream_index(StreamIndex si) {
    // TODO: check whether the stream index is valid
    stream_index = si;
    return *this;
  }

  inline OpMeta& set_device_group(const DeviceGroup& g) {
    device_group = g;
    return *this;
  }

  inline OpMeta& set_device_group(DeviceGroup&& group) {
    device_group = std::move(group);
    return *this;
  }

  inline OpMeta& set_extra_deps(const TensorList& deps) {
    extra_deps = deps;
    return *this;
  }

  inline OpMeta& set_extra_deps(TensorList&& deps) {
    extra_deps = std::move(deps);
    return *this;
  }

  inline OpMeta& set_is_deduce_states(bool deduce_states) {
    is_deduce_states = deduce_states;
    return *this;
  }

  inline OpMeta& set(const OpMeta& other) {
    operator=(other);
    return *this;
  }

  inline OpMeta& set(OpMeta&& other) {
    operator=(std::move(other));
    return *this;
  }

  static OpMeta Merge(const OpMeta& base_meta, const OpMeta& new_meta) {
    OpMeta ret = base_meta;
    if (!new_meta.name.empty())
      ret.set_name(new_meta.name);
    if (new_meta.stream_index != kUndeterminedStream)
      ret.set_stream_index(new_meta.stream_index);
    if (!new_meta.device_group.empty())
      ret.set_device_group(new_meta.device_group);
    if (!new_meta.extra_deps.empty())
      ret.set_extra_deps(new_meta.extra_deps);
    return ret;
  }

  OpName name;
  StreamIndex stream_index{kUndeterminedStream};
  DeviceGroup device_group;
  TensorList extra_deps;
  bool is_deduce_states{true};
};

std::ostream& operator<<(std::ostream&, const OpMeta&);

} // namespace autograd
} // namespace hetu

namespace std {
inline std::string to_string(const hetu::autograd::OpMeta& meta) {
  std::ostringstream os;
  os << meta;
  return os.str();
}
} // namespace std

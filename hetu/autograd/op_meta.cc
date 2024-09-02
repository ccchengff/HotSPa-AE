#include "hetu/autograd/op_meta.h"
#include "hetu/autograd/tensor.h"
#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

std::ostream& operator<<(std::ostream& os, const OpMeta& meta) {
  os << "{";
  bool first = true;
  if (meta.stream_index != kUndeterminedStream) {
    if (!first)
      os << ", ";
    os << "stream_index=" << meta.stream_index;
    first = false;
  }
  if (!meta.device_group.empty()) {
    if (!first)
      os << ", ";
    os << "device_group=" << meta.device_group;
    first = false;
  }
  if (!meta.extra_deps.empty()) {
    if (!first)
      os << ", ";
    os << "extra_deps=" << meta.extra_deps;
    first = false;
  }
  if (!meta.name.empty()) {
    if (!first)
      os << ", ";
    os << "name=" << meta.name;
    first = false;
  }
  os << "}";
  return os;
}

} // namespace autograd
} // namespace hetu

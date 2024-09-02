#include "hetu/core/binary_type.h"

namespace hetu {

std::string BinaryType2Str(const BinaryType& binary_type) {
  switch (binary_type) {
    case BinaryType::ADD: return "ADD";
    case BinaryType::SUB: return "SUB";
    case BinaryType::MUL: return "MUL";
    case BinaryType::DIV: return "DIV";
    case BinaryType::MOD: return "MOD";
    default:
      HT_VALUE_ERROR << "Unknown binary type: "
                     << static_cast<int32_t>(binary_type);
      __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const BinaryType& binary_type) {
  os << BinaryType2Str(binary_type);
  return os;
}

} // namespace hetu

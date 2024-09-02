#include "hetu/core/reduction_type.h"

namespace hetu {

std::string ReductionType2Str(const ReductionType& red_type) {
  switch (red_type) {
    case ReductionType::SUM: return "sum";
    case ReductionType::MEAN: return "mean";
    case ReductionType::PROD: return "prod";
    case ReductionType::MAX: return "max";
    case ReductionType::MIN: return "min";
    case ReductionType::NONE: return "none";
    default:
      HT_VALUE_ERROR << "Unknown reduction type: "
                     << static_cast<int32_t>(red_type);
      __builtin_unreachable();
  }
}

ReductionType Str2ReductionType(const std::string& type) {
  if (type == "sum") 
    return ReductionType::SUM;
  if (type == "mean")
    return ReductionType::MEAN;
  if (type == "prod")
    return ReductionType::PROD;
  if (type == "max")
    return ReductionType::MAX;
  if (type == "min")
    return ReductionType::MIN;
  if (type == "none")
    return ReductionType::NONE;
  HT_VALUE_ERROR << "Unknown reduction type: " << type;
  __builtin_unreachable();
}

std::ostream& operator<<(std::ostream& os, const ReductionType& red_type) {
  os << ReductionType2Str(red_type);
  return os;
}

} // namespace hetu

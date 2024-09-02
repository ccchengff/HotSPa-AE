#include "hetu/core/bfloat16.h"
#include <iostream>
namespace hetu {
std::ostream& operator<<(std::ostream& out, const bfloat16& value) {
  out << (float)value;
  return out;
}
} //namespace hetu

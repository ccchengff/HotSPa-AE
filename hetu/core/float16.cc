#include "hetu/core/float16.h"
#include <iostream>
namespace hetu {
std::ostream& operator<<(std::ostream& out, const float16& value) {
  out << (float)value;
  return out;
}
} //namespace hetu

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>

namespace hetu {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << '[';
  if (!vec.empty()) {
    os << vec[0];
    for (size_t i = 1; i < vec.size(); i++)
      os << ',' << vec[i];
  }
  os << ']';
  return os;
}

template <template <typename...> class Set, typename T>
std::ostream& _set_ostreaming(std::ostream& os, const Set<T>& set) {
  os << '{';
  if (!set.empty()) {
    auto it = set.begin();
    os << *it;
    for (it++; it != set.end(); it++) {
      os << ',' << *it;
    }
  }
  os << '}';
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
  return _set_ostreaming(os, set);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& set) {
  return _set_ostreaming(os, set);
}

template <template <typename...> class Map, typename K, typename V>
std::ostream& _map_ostreaming(std::ostream& os, const Map<K, V>& map) {
  os << '{';
  if (!map.empty()) {
    auto it = map.begin();
    os << it->first << ':' << it->second;
    for (it++; it != map.end(); it++) {
      os << ',' << it->first << ':' << it->second;
    }
  }
  os << '}';
  return os;
}

template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::map<K, V>& map) {
  return _map_ostreaming(os, map);
}

template <typename K, typename V>
std::ostream& operator<<(std::ostream& os,
                         const std::unordered_map<K, V>& map) {
  return _map_ostreaming(os, map);
}

} // namespace hetu

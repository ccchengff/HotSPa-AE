#pragma once

#include "hetu/common/macros.h"
#include "hetu/utils/optional.h"
#include <stack>

namespace hetu {

template <typename T>
class ContextManager {
 public:
  ContextManager(optional<T> default_value = nullopt)
  : _default_value(default_value) {}

  optional<T> peek() {
    return !_stack.empty() ? _stack.top() : _default_value;
  }

  inline void push(const T& value) { _stack.push(optional<T>(value)); }

  inline void push(T&& value) { _stack.push(optional<T>(value)); }

  inline optional<T> pop() {
    HT_RUNTIME_ERROR_IF(_stack.empty()) 
      << "Cannot pop from an empty context stack";
    auto ret = _stack.top();
    _stack.pop();
    return ret;
  }

 private:
  std::stack<optional<T>> _stack;
  optional<T> _default_value;
};

} // namespace hetu


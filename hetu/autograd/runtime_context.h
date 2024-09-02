#pragma once

#include "hetu/common/macros.h"
#include "hetu/utils/context_store.h"
#include <memory>

namespace hetu {
namespace autograd {

// forward declarition of OpId
using FwOpId = int64_t;

class OpContext : public ContextStore {
 public:
  OpContext(FwOpId op_id) : ContextStore(), _op_id(op_id) {}

  FwOpId op_id() const noexcept {
    return _op_id;
  }

 private:
  const FwOpId _op_id;
};

class RuntimeContext {
 public:
  RuntimeContext(size_t max_op_id_hint = 4096) {
    _op_ctxs.resize(max_op_id_hint);
  }

  OpContext& get_op_ctx(FwOpId op_id) {
    if (_op_ctxs.size() < static_cast<size_t>(op_id) + 1)
      _op_ctxs.resize(op_id + 1);
    if (_op_ctxs[op_id] == nullptr) {
      _op_ctxs[op_id].reset(new OpContext(op_id));
    }
    return *_op_ctxs[op_id];
  }

 private:
  std::vector<std::unique_ptr<OpContext>> _op_ctxs;
};

} // namespace autograd
} // namespace hetu

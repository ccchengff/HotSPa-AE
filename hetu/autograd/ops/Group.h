#pragma once

#include "hetu/autograd/operator.h"
#include <numeric>

namespace hetu {
namespace autograd {

namespace {
inline OpMeta _MergeDeps(const TensorList& deps, const OpMeta& op_meta) {
  OpMeta ret = op_meta;
  auto& extra_deps = ret.extra_deps;
  extra_deps.reserve(deps.size() + extra_deps.size());
  extra_deps.insert(extra_deps.end(), deps.begin(), deps.end());
  return ret;
}
} // namespace

class GroupOpDef;
class GroupOp;

class GroupOpDef : public OperatorDef {
 private:
  friend class GroupOp;
  struct constrcutor_access_key {};

 public:
  GroupOpDef(const constrcutor_access_key&,
             const TensorList& deps = TensorList(),
             const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(GroupOp), TensorList(), _MergeDeps(deps, op_meta)) {}

  const TensorList& get_in_dep_linkers() const {
    return _extra_in_dep_linkers;
  }

  uint64_t op_indicator() const noexcept {
    return GROUP_OP;
  }

 protected:
  void DoCompute(const NDArrayList& deps, NDArrayList& outputs,
                 RuntimeContext& ctx) override {}
};

class GroupOp final : public OpWrapper<GroupOpDef> {
 public:
  GroupOp(const TensorList& deps = TensorList(),
          const OpMeta& op_meta = OpMeta())
  : OpWrapper<GroupOpDef>(make_ptr<GroupOpDef>(
      GroupOpDef::constrcutor_access_key(), deps, op_meta)) {}
};

} // namespace autograd
} // namespace hetu

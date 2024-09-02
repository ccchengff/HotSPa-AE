#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class LossOpImpl;
class LossGradientOpImpl;

class LossOpImpl : public OpInterface {
 protected:
  LossOpImpl(OpType&& type, ReductionType reduction = kMEAN)
  : OpInterface(std::move(type)), _reduction(reduction) {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
  }

  ReductionType _reduction;  

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  uint64_t op_indicator() const noexcept override {
    return LOSS_OP;
  }
  
  ReductionType reduction() const {
    return _reduction;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LossOpImpl&>(rhs);
      return reduction() == rhs_.reduction();
    }
    return false;
  }
};

class LossGradientOpImpl : public LossOpImpl {
 protected:
  LossGradientOpImpl(OpType&& type, ReductionType reduction = kMEAN)
  : LossOpImpl(std::move(type), reduction) {}

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  uint64_t op_indicator() const noexcept override {
    return LOSS_GRADIENT_OP;
  } 

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LossGradientOpImpl&>(rhs);
      return reduction() == rhs_.reduction();
    }
    return false;
  }
};

} // namespace graph
} // namespace hetu
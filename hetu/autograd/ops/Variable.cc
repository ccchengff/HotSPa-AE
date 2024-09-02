#include "hetu/autograd/ops/Variable.h"

namespace hetu {
namespace autograd {

bool VariableOpDef::DoPlaceToLocalDevice(const Device& placement,
                                         StreamIndex stream_id) {
  auto dtype = _outputs[0]->dtype();
  if (!_data.is_defined()) {
    _data = NDArray::empty(_shape, placement, dtype);
  } else if (_data->device() != placement || _data->dtype() != dtype) {
    _data = NDArray::to(_data, placement, dtype, kBlockingStream);
  }
  if (_init != nullptr) {
    if (!_inited) {
        _init->Init(_data, 0, kBlockingStream);
        _inited = 1;
    }
  }
  return OperatorDef::DoPlaceToLocalDevice(placement, stream_id);
}

TensorList PlaceholderBaseOpDef::DoGradient(const TensorList& grad_outputs) {
  return {Tensor()};
}

HTShapeList
PlaceholderBaseOpDef::DoInferShape(const HTShapeList& input_shapes) {
  HT_RUNTIME_ERROR_IF(_shape.empty())
    << "Placeholder " << name() << " should be provided by feed_dict";
  return {_shape};
}

void VariableOpDef::reset_initializer(const Initializer& init) {
  _init.reset(init.copy());
  _inited = 0;
  if (!_placement.is_undetermined()) {
    _init->Init(_data, 0, kBlockingStream);
    _inited = 1;
  }
}

void VariableOpDef::reset_data(const NDArray& data) {
  _data = NDArray::copy(data, kBlockingStream, _data);
  if (_placement.is_undetermined()) {
    _init = nullptr;
  }
}

} // namespace autograd
} // namespace hetu

#include "hetu/autograd/ops/DataTransfer.h"
#include "hetu/autograd/ops/kernel_links.h"
#include "hetu/impl/stream/CUDAStream.h"

namespace hetu {
namespace autograd {

DataH2DOpDef::DataH2DOpDef(const constrcutor_access_key&, Tensor input,
                           const Device& device, const OpMeta& op_meta)
: OperatorDef(quote(DataH2DOp), {input},
              OpMeta().set(op_meta).set_device_group(DeviceGroup({device}))) {
  HT_ASSERT(device_group().num_devices() == 1 &&
            device_group().get(0).is_cuda())
    << "Failed to construct the \"" << type() << "\" operation "
    << "(with name \"" << name() << "\"): "
    << "The device group must be containing a single CUDA device. "
    << "Got " << device_group().devices() << ".";
  AddOutput(NDArrayMeta().set(input->meta()).set_device(device_group().get(0)));
  DoDeduceStates();
}

bool DataH2DOpDef::DoPlaceToLocalDevice(const Device& placement,
                                        StreamIndex stream_id) {
  if (!placement.is_cuda())
    return false;

  HT_ASSERT(_inputs[0]->placement().is_cpu())
    << "Failed to place the \"" << _type << "\" operation "
    << "(with name \"" << name() << "\"): "
    << "The input \"" << _inputs[0]->name() << "\" must be placed on "
    << "a host device. Got " << _inputs[0]->placement() << ".";
  _placement = placement;
  _stream = Stream(_placement, stream_id);
  _start = std::make_shared<hetu::impl::CUDAEvent>(_placement);
  _stop = std::make_shared<hetu::impl::CUDAEvent>(_placement);
  _outputs[0]->set_placement(placement);
  return true;
}

void DataH2DOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  hetu::impl::DataTransferCuda(inputs.at(0), outputs.at(0), stream());
}

TensorList DataH2DOpDef::DoGradient(const TensorList& grad_outputs) {
  return {
    DataD2HOp(
      grad_outputs.at(0),
      grad_op_meta().set_device_group(DeviceGroup()).set_name(grad_name()))
      ->output(0)};
}

HTShapeList DataH2DOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

DataD2HOpDef::DataD2HOpDef(const constrcutor_access_key&, Tensor input,
                           const OpMeta& op_meta)
: OperatorDef(quote(DataD2HOp), {input},
              OpMeta().set(op_meta).set_device_group(DeviceGroup({kCPU}))) {
  HT_ASSERT(device_group().num_devices() == 1 && device_group().get(0).is_cpu())
    << "Failed to construct the \"" << type() << "\" operation "
    << "(with name \"" << name() << "\"): "
    << "the device group must be containing a single host device. "
    << "Got " << device_group().devices() << ".";
  AddOutput(NDArrayMeta().set(input->meta()).set_device(device_group().get(0)));
  DoDeduceStates();
}

bool DataD2HOpDef::DoPlaceToLocalDevice(const Device& placement,
                                        StreamIndex stream_id) {
  if (!placement.is_cpu())
    return false;

  HT_ASSERT(_inputs[0]->placement().is_cuda())
    << "Failed to place the \"" << type() << "\" operation "
    << "(with name \"" << name() << "\"): "
    << "The input \"" << _inputs[0]->name() << "\" must be placed on "
    << "a CUDA device. Got " << _inputs[0]->placement() << ".";
  _placement = placement;
  _stream = Stream(_inputs[0]->placement(), stream_id);
  _start = std::make_shared<hetu::impl::CUDAEvent>(_inputs[0]->placement());
  _stop = std::make_shared<hetu::impl::CUDAEvent>(_inputs[0]->placement());
  _outputs[0]->set_placement(placement);
  return true;
}

void DataD2HOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) {
  hetu::impl::DataTransferCuda(inputs.at(0), outputs.at(0), stream());
}

TensorList DataD2HOpDef::DoGradient(const TensorList& grad_outputs) {
  return {
    DataH2DOp(
      grad_outputs.at(0), grad_outputs.at(0)->placement(),
      grad_op_meta().set_device_group(DeviceGroup()).set_name(grad_name()))
      ->output(0)};
}

HTShapeList DataD2HOpDef::DoInferShape(const HTShapeList& input_shapes) {
  return {input_shapes.at(0)};
}

} // namespace autograd
} // namespace hetu

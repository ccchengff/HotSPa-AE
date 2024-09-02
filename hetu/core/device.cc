#include "hetu/core/device.h"

namespace hetu {

std::vector<std::pair<bool, cudaDeviceProp>> Device::CUDAInit() {
  std::vector<std::pair<bool, cudaDeviceProp>> dprops;
  for (int i = 0; i < HT_MAX_DEVICE_INDEX; ++i) {
    dprops.emplace_back(false, cudaDeviceProp());
  }
  return dprops;
}
std::vector<std::pair<bool, cudaDeviceProp>> Device::_dprops = Device::CUDAInit();

std::string DeviceType2Str(const DeviceType& t) {
  switch (t) {
    case kCPU: return "cpu";
    case kCUDA: return "cuda";
    case kUndeterminedDevice: return "undetermined";
    default:
      HT_VALUE_ERROR << "Unknown device type: " << static_cast<int16_t>(t);
      __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const DeviceType& t) {
  os << DeviceType2Str(t);
  return os;
}

DeviceType Str2DeviceType(const std::string& type) {
  if (type == "cpu")
    return kCPU;
  if (type == "cuda")
    return kCUDA;
  if (type == "undetermined")
    return kUndeterminedDevice;
  HT_VALUE_ERROR << "Unknown device type: " << type;
  __builtin_unreachable();
}

std::tuple<DeviceType, DeviceIndex, std::string>
Str2DeviceTypeAndIndexAndHostname(const std::string& device) {
  DeviceType type;
  DeviceIndex index = 0U;
  std::string hostname;
  size_t backslash = device.rfind(Device::BACK_SLASH);
  size_t offset = 0;
  if (backslash != std::string::npos) {
    hostname = device.substr(0, backslash);
    offset = backslash + 1;
  }
  size_t colon = device.find(Device::COLON, offset);
  if (colon == std::string::npos) {
    type = Str2DeviceType(offset == 0 ? device : device.substr(offset));
  } else if (colon + 1 == device.length()) {
    type = Str2DeviceType(device.substr(offset, colon - offset));
  } else {
    type = Str2DeviceType(device.substr(offset, colon - offset));
    auto index_str = device.substr(colon + 1);
    bool is_digits = index_str.end() ==
      std::find_if(index_str.begin(), index_str.end(),
                   [](auto c) { return !std::isdigit(c); });
    HT_VALUE_ERROR_IF(!is_digits) << "Cannot parse device: " << device;
    index = std::stoi(index_str);
  }
  return {type, index, hostname};
}

Device::Device(const std::string& device, uint8_t multiplex) {
  auto t = Str2DeviceTypeAndIndexAndHostname(device);
  _init(std::get<0>(t), std::get<1>(t), std::get<2>(t), multiplex);
}

std::string Device::compat_string() const {
  std::ostringstream os;
  if (!local())
    os << hostname() << Device::BACK_SLASH;
  os << type();
  if (index() != 0)
    os << Device::COLON << static_cast<int>(index());
  return os.str();
}

std::ostream& operator<<(std::ostream& os, const Device& device) {
  os << "device(type=" << device.type();
  if (device.is_cuda())
    os << ", index=" << static_cast<int>(device.index());
  if (!device.local())
    os << ", hostname=" << device.hostname();
  if (device.multiplex())
    os << ", multiplex=" << static_cast<int>(device.multiplex());
  os << ')';
  return os;
}

std::ostream& operator<<(std::ostream& os, const DeviceGroup& device_group) {
  os << "DeviceGroup(" << device_group.devices() << ")";
  return os;
}

} // namespace hetu
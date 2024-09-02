#pragma once

#include "hetu/autograd/common.h"
#include <functional>

namespace hetu {
namespace autograd {
  
class DistributedStates {
 public:
  // 1. Tensor创建时该属性默认为空, 需要后续赋值 
  DistributedStates() : _device_num(-1), _states({}), _order({}) {};
  // 2. 在Tensor创建时就直接定义好切分状态, 此时直接传递placement group即可
  DistributedStates(DeviceGroup& placement_group, const std::unordered_map<int32_t, int32_t>& states, 
                    const std::vector<int32_t>& order = {}) { // states/order都是const的引用, 因此未来要update的话只能先copy创建一个新对象再对其修改
    _placement_group = placement_group;
    _device_num = placement_group.num_devices();
    set_states(states);
    set_order(order); 
  }
  // 3. 作为单独的分布式属性存在, 可用于指定要转换的目前切分状态, 此时暂时只需要赋值device_num, 而placement_group要与Tensor绑定后赋值
  DistributedStates(int32_t device_num, const std::unordered_map<int32_t, int32_t>& states, 
                    const std::vector<int32_t>& order = {}) {
    _placement_group = DeviceGroup(); // 空的device group, 在和tensor binding时需要填充
    _device_num = device_num;
    set_states(states);
    set_order(order); 
  }

  // 假设dp被包含在distributed attributes里的states[0], 则_device_num实际上就等于device placement_group的size
  // 理论上除了pp把不同layer的op切分到几组不同的device group之外, dp和tp中所有op都是共享同一组device group的
  // 只有在和Tensor绑定时才会调用该函数来赋值placement group
  void set_placement_group(const DeviceGroup& placement_group);

  // 主要用来确定local device在device group中的index, 以便定位该device的部分tensor在global tensor中的位置
  void set_placement(const Device& placement);

  // 用来给placeholder_op/variable_op这类input/variable tensor赋值states/order用, 其中placement等信息已经在map to devices的时候赋值过了
  // 在外面更新states/order推荐使用set_distributed_states, 同时赋值states和order, 保证其一致性
  void set_distributed_states(const DistributedStates& dst_distributed_states);

  // states/order是否已经被成功赋值, 理论上通过构造函数/set_distributed_states进行赋值的, 正确性是已经保证了的, 这里只需要验证有没有
  bool is_valid();

  std::unordered_map<int32_t, int32_t> get_states() {
    return _states;
  }

  std::vector<int32_t> get_order() {
    return _order;
  }

  DeviceGroup get_placement_group() {
    return _placement_group;
  }

  Device get_placement() {
    return _placement;
  }

  int32_t get_placement_index() {
    return _placement_group.get_index(_placement);
  }

  int32_t get_device_num() {
    return _device_num;
  }

  std::unordered_map<int32_t, int32_t> combine_states(std::pair<std::vector<int32_t>, int32_t>& src2dst);
  std::vector<int32_t> combine_order(std::pair<std::vector<int32_t>, int32_t>& src2dst);
  bool equal_states_and_order(std::unordered_map<int32_t, int32_t>& states1, std::vector<int32_t>& order1,
                              std::unordered_map<int32_t, int32_t>& states2, std::vector<int32_t>& order2);
  bool check_equal(DistributedStates& dst_distributed_states);
  bool check_max_dim(int32_t max_dim);
  bool check_pure_duplicate();    
  bool check_combine(DistributedStates& dst_distributed_states, std::pair<std::vector<int32_t>, int32_t>& src2dst);

  std::unordered_map<int32_t, int32_t> reduce_states(int dim);
  std::vector<int32_t> reduce_order(int dim);
  bool check_reduce_dim(DistributedStates& dst_distributed_states, int dim);

  bool check_allreduce(DistributedStates& dst_distributed_states);
  bool check_allgather(DistributedStates& dst_distributed_states);
  bool check_reducescatter(DistributedStates& dst_distributed_states);
  bool check_boradcast(DistributedStates& dst_distributed_states);
  bool check_reduce(DistributedStates& dst_distributed_states);  

  int32_t get_dim(int32_t index);
  std::vector<int32_t> get_loop_sizes();
  std::unordered_map<int32_t, int32_t> map_device_to_state_index(int32_t device_index); // for single device
  std::string ds_info();

 protected:
  // 同时赋值states和order, 保证其一致性
  void set_states(const std::unordered_map<int32_t, int32_t>& states);

  // 同时赋值states和order, 保证其一致性
  void set_order(const std::vector<int32_t>& order);

  // partial和duplicate的key要求必须存在, 值可以是1表示没有; 其余dimension只有存在切分的时候key:value才会存在于states中
  std::unordered_map<int32_t, int32_t> _states; // {dimension: split_num}, {-2: partial, -1: duplicate, 0~n-1: dimension}
  std::vector<int32_t> _order; // for device mapping
  DeviceGroup _placement_group; // 在和Tensor binding的时候必须设置, 否则可以为空
  int32_t _device_num; // 如果_placement_group为空, 则_device_num必须设置
  Device _placement;
  bool _is_distributed; // if false, do not deduce states 
};

} // namespace autograd
} // namespace hetu
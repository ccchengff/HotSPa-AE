#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/autocast/autocast.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"

namespace hetu {
namespace graph {

// changing parallel plan
static size_t change_parallel_test_case = 0;

Operator& DefineAndRunGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                         TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  // HT_LOG_TRACE << name() << " make op: " << op_meta.name;
  auto& op = MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
  // record the ops that have an explicit device group setting
  // which will then used to deduce the pp stages
  if (op->device_group_hierarchy().size() == NUM_STRATEGY) {
    _ops_with_device_group_hierarchy.emplace_back(op);
  } 
  return _op_indexing[op->id()];
}

void DefineAndRunGraph::ResetVariableDataInner(const Tensor& tensor,
                                               const Initializer& init) {
  // Mark an add-on initializer.
  _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  if (_is_active) {
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
    auto it = tensor_to_exec_tensor_mapping.find(tensor->id());
    if (it != tensor_to_exec_tensor_mapping.end()) {
      // The op has been instantiated in the current active graph. Also let the executable graph reset it.
      Graph::ResetVariableData(it->second, init);
    }
  }
}

NDArray DefineAndRunGraph::GetDetachedVariableDataInner(const Tensor& tensor) {
  if (_is_active) {
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
    auto it_1 = tensor_to_exec_tensor_mapping.find(tensor->id());
    if (it_1 == tensor_to_exec_tensor_mapping.end()) {
      // The tensor is not in current active exec graph.
      // Question: store the data on different devices? For now, store all on CPU and return.
      auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      auto it_2 = _add_on_inits.find(tensor->id());
      // Note _add_on_inits has a higher priority than the original tensor initializer.
      if (it_2 != _add_on_inits.end()) {
        HT_LOG_TRACE << "The data is reset, but not in current active exec graph, "
          << "so we get the data of the variable from the DefineAndRun graph.";
        it_2->second->Init(ret);
      } else {
        HT_LOG_TRACE << "The data is not in current active exec graph, " 
          << "so we get the data of the variable from its initializer.";
        if (tensor->has_distributed_states())
          dynamic_cast<ParallelVariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);
        else
          dynamic_cast<VariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);  
      }
      Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
      stream.Sync();
      return ret;
    } else {
      // The op has been instantiated in the current active graph. Let the executable graph handle it.
      if (!it_1->second->producer()->placement_group_union().has(impl::comm::GetLocalDevice())) {
        HT_LOG_TRACE << "The data is not locate at local executable graph, return an empty NDArray.";
        return NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      }
      auto ret = Graph::GetDetachedVariableData(it_1->second);
      Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
      stream.Sync();
      return ret;
    }  
  } else {
    auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
    auto it = _add_on_inits.find(tensor->id());
    // Note _add_on_inits has a higher priority than the original tensor initializer.
    if (it != _add_on_inits.end()) {
      HT_LOG_TRACE << "No active exec graph yet. The data is reset, " 
        << "so we get the data of the variable from the DefineAndRun graph.";
      it->second->Init(ret);
    } else {
      HT_LOG_TRACE << "No active exec graph yet, "
        << "so we get the data of the variable from its initializer.";
      if (tensor->has_distributed_states())
        dynamic_cast<ParallelVariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);
      else
        dynamic_cast<VariableOpImpl&>(tensor->producer()->body()).initializer().Init(ret);  
    }
    Stream stream(Device(kCPU), NDArray::DEFAULT_STREAM);
    stream.Sync();
    return ret;
  }
}

DeviceGroupUnion DefineAndRunGraph::GetVariableDeviceGroupUnionInner(const Tensor& tensor) {
  auto& device_group_union = tensor->producer()->device_group_union();
  HT_RUNTIME_ERROR_IF(device_group_union.size() == 0) << "You are getting an empty device group union, please ensure you have set "
    << tensor->producer() << " a device group hierarchy before!";
  return device_group_union;
}

void DefineAndRunGraph::MergeGraph(DefineAndRunGraph& another_graph) {
  HT_ASSERT(_op_indexing.size() == another_graph._op_indexing.size())
    << "two graph op indexing should be aligned";
  NUM_STRATEGY += another_graph.NUM_STRATEGY;
  // workaround
  // 按照op index进行对齐
  for (auto& kv : _op_indexing) {
    OpId op_id =  kv.first;
    Operator& op = kv.second;
    auto it = another_graph._op_indexing.find(op_id);
    HT_ASSERT(it != another_graph._op_indexing.end())
      << "cannot find " << op << " in the graph to merge";
    Operator& another_op = it->second;
    op->MergeStrategy(another_op);
  }
}

// 推导define graph在cur_strategy_id下的pipeline构造
void DefineAndRunGraph::DeducePipeline(size_t cur_strategy_id, int32_t pipeline_num) {
  auto old_strategy_id = CUR_STRATEGY_ID;
  CUR_STRATEGY_ID = cur_strategy_id;
  std::unordered_map<Device, int32_t> device_to_pipeline_idx_map;
  std::vector<DeviceGroupList> pipelines(pipeline_num);
  int32_t total_p2pline = -1;
  int32_t total_pipeline = std::numeric_limits<int32_t>::max();
  // 得到所有pipeline
  // 以dp2tp2pp2为例
  // bias的dup为4但weight的dup为2
  // 那么pipeline条数为2而不是4
  // 因此实际上其取决于最小的dup
  // 其本质是除了dup外就是split而split一定不能使用不同的feed dict的data
  // 因此做split的参数最终一定要求是在同一个pipeline中
  for (const auto& op : _ops_with_device_group_hierarchy) {
    // deduce pipeline的stages时只需要用到模型的parameters
    if (_parameter_ops.find(op->id()) == _parameter_ops.end()) {
      continue;
    }
    auto& dg_union = op->device_group_union();
    auto& ds_union = op->output(0)->cur_ds_union();
    // HT_LOG_INFO << op << " device group union: " << dg_union << " and ds union: " << ds_union.ds_union_info();
    HT_ASSERT(dg_union.size() != 0 && ds_union.size() != 0 && dg_union.size() == ds_union.size())
      << "dg union & ds union of " << op << " shouldn't be empty and should have the same size";
    int32_t union_size = static_cast<int32_t>(dg_union.size());
    int32_t dup_size = -1;
    bool is_hetero = union_size > 1;
    // hetero情况
    if (is_hetero) {
      HT_ASSERT(ds_union.hetero_dim() == -1)
        << "Now onlt support param hetero on dup (-1)";
      for (size_t i = 0; i < union_size; i++) {
        auto& ds = ds_union.get(i);
        auto ds_dup = ds.states(-1);
        if (ds_dup != 1) {
          HT_LOG_WARN_IF(ds.order(0) != -1) 
            << op << " is a parameter, which is suggested to put dup at the first position in the ds order sequence"
            << ", but now the ds is: " << ds.ds_info();
        }
      }
      dup_size = union_size;
    } 
    // homo情况
    else {
      dup_size = ds_union.get(0).get_dim(-1);
    }
    HT_ASSERT(dup_size % pipeline_num == 0 && dup_size >= pipeline_num)
      << "dup size should be divided by the number of pipelines";
    // 相邻merge_ratio个dup算在一个pipeline里
    size_t merge_ratio = dup_size / pipeline_num;
    std::vector<std::vector<Device>> stages(pipeline_num);
    // 不存在异构dp
    // 即union中只有一个
    // 默认在dup维度均分几条pipeline
    if (!is_hetero) {
      auto& ds = ds_union.get(0);
      auto& device_group = dg_union.get(0);
      for (int32_t i = 0; i < device_group.num_devices(); i++) {
        auto state_index = ds.map_device_to_state_index(i);
        int32_t dup_idx = dup_size == 1 ? 0 : state_index[-1];
        int32_t pipeline_idx = dup_idx / merge_ratio;
        // 记录device到pipeline的映射
        if (device_to_pipeline_idx_map.find(device_group.get(i)) != device_to_pipeline_idx_map.end()) {
          HT_ASSERT(device_to_pipeline_idx_map[device_group.get(i)] == pipeline_idx)
            << "device " << device_group.get(i) << " is in two pipelines, which will cause chaos"
            << ", the existing pipeline is " << device_to_pipeline_idx_map[device_group.get(i)]
            << " and the new pipeline is " << pipeline_idx; 
        } else {
          device_to_pipeline_idx_map[device_group.get(i)] = pipeline_idx;
        }
        // 添加device到新的stage
        stages[pipeline_idx].emplace_back(device_group.get(i));
      }
    }
    // 存在异构
    // union中每个device group本质上就是一个stage
    // 当然还要考虑相邻dup进行merge的情形
    else {
      for (size_t i = 0; i < pipeline_num; i++) {
        for (size_t j = 0; j < merge_ratio; j++) {
          auto& device_group = dg_union.get(i * merge_ratio + j);
          for (const auto& device : device_group.devices()) {
            // 记录device到pipeline的映射
            if (device_to_pipeline_idx_map.find(device) != device_to_pipeline_idx_map.end()) {
              HT_ASSERT(device_to_pipeline_idx_map[device] == i)
                << "device " << device << " is in two pipelines, which will cause chaos"
                << ", the existing pipeline is " << device_to_pipeline_idx_map[device]
                << " and the new pipeline is " << i; 
            } else {
              device_to_pipeline_idx_map[device] = i;
            }
            // 添加device到新的stage
            stages[i].emplace_back(device);  
          }
        }
      }
    }
    // 添加各个stage到各个pipeline
    for (size_t i = 0; i < pipeline_num; i++) {
      if (!stages[i].empty()) {
        auto cur_stage_device_group = DeviceGroup(stages[i]);
        int32_t cur_pipeline_len = static_cast<int32_t>(pipelines[i].size());
        for (int32_t j = 0; j < cur_pipeline_len; j++) {
          if (j != cur_pipeline_len - 1) {
            HT_ASSERT(cur_stage_device_group != pipelines[i][j])
              << "Duplicate stage in a single pipeline";
          }
        }
        if (cur_pipeline_len == 0 || pipelines[i][cur_pipeline_len - 1] != cur_stage_device_group) {
          pipelines[i].emplace_back(std::move(cur_stage_device_group));
        }
      }
    }
  }
  // 记录当前strategy下的device到pipeline映射
  for (const auto& kv : device_to_pipeline_idx_map) {
    const auto& device = kv.first;
    auto pipeline_idx = kv.second;
    _multi_pipeline_maps[CUR_STRATEGY_ID][device] = pipelines[pipeline_idx];
  }
  // workaround
  // 获取当前device推荐的hetero id
  auto it = device_to_pipeline_idx_map.find(hetu::impl::comm::GetLocalDevice());
  if (it != device_to_pipeline_idx_map.end()) {
    SUGGESTED_HETERO_ID = it->second;
  }
  CUR_STRATEGY_ID = old_strategy_id;
}

// 推导define graph的shape plan
// 以及exec graph的exec shape plan
// 请注意二者的区别
// 前者是用来进行plan匹配的（虽然目前feed dict固定实际上只用记录feed dict的shape）
// 后者是实际exec graph执行时runtime ctx要用到的
void DefineAndRunGraph::DeduceShapePlan(ExecGraphPlan& exec_graph_plan,
                                        const FeedDict& feed_dict,
                                        Tensor2ShapeMap& feed_dict_shape) {
  // *the logic of inferring the very first shape plan is in Instantiate()
  // that is because MakeOp can handle most of the cases automatically
  // InferShapePlan just aims to expand the shape plan pool for the data packing setting
  auto local_device = hetu::impl::comm::GetLocalDevice(); // debug use
  Tensor2ShapeMap shape_plan;
  Tensor2ShapeMap exec_shape_plan;
  RuntimeContext runtime_ctx{};
  // 扫描global topo并推导新的shape plan
  // *这里的shape plan是define graph上的
  for (auto& op_ref : exec_graph_plan.global_topo) {
    auto& op = op_ref.get();
    // 设置placeholder（也有可能是中间的算子——具体要看feed_dict喂的是什么算子）的symbolic shape
    bool handle_feed_dict_op = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
      auto it = feed_dict.find(tensor->id());
      if (it != feed_dict.end()) {
        if (tensor->symbolic() && is_SyShape_leaf(tensor->symbolic_shape())) {
          tensor->set_symbolic_shape(feed_dict_shape[tensor->id()]);
          HT_LOG_DEBUG << local_device << ": set symbolic shape of " << op 
            << " feed_dict tensor to " << feed_dict_shape[tensor->id()];
        }
        shape_plan[tensor->id()] = feed_dict_shape[tensor->id()];
        return true;
      }
      return false;
    });
    if (handle_feed_dict_op) {
      continue;
    }
    HTShapeList input_shapes;
    input_shapes.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      auto it = shape_plan.find(input->id());
      HT_ASSERT(it != shape_plan.end()) 
        << "Something wrong, can't find the input shape from the current shape plan!";
      input_shapes.push_back(it->second);
    }
    auto it = exec_graph_plan.op_to_exec_op_mapping.find(op->id());
    HT_ASSERT(it != exec_graph_plan.op_to_exec_op_mapping.end())
      << op << " doesn't have an exec version";
    auto& exec_op = it->second;
    // 使用exec op的InferShape而不是op的InferShape
    // 因为exec op已经具有placement group union
    // 因此可以得到local device对应的ds
    HTShapeList exec_output_shapes = exec_op->InferShape(input_shapes, runtime_ctx);
    auto exec_output_shapes_size = exec_output_shapes.size();
    for (size_t i = 0; i < exec_output_shapes_size; i++) {
      // 设置symbolic shape叶子节点的shape
      // 其相关联的非叶子的symbolic shape可以直接由计算链条获得新的shape
      if (exec_op->output(i)->symbolic()) {
        HT_LOG_TRACE << local_device << ": op " << exec_op 
          << " output " << i << " has " << exec_op->output(i)->symbolic_shape();
        if (is_SyShape_leaf(exec_op->output(i)->symbolic_shape())) {
          exec_op->output(i)->set_symbolic_shape(exec_output_shapes[i]);
          HT_LOG_TRACE << local_device << ": set symbolic shape of " << op 
            << " output " << i << " to " << exec_output_shapes[i];
        }
      }
      HT_LOG_TRACE << local_device << ": " << op->output(i) << " shape " << exec_output_shapes[i];
      auto it = shape_plan.find(op->output(i)->id());
      HT_ASSERT(it == shape_plan.end()) 
        << "Something wrong, the output shape should't exist in the current shape plan";
      shape_plan.insert(std::make_pair(op->output(i)->id(), std::move(exec_output_shapes[i]))); // move constructor
    }
  }
  // define graph中已经推导得到的shape plan
  // 赋值给exec graph的exec shape plan
  for (const auto& kv : exec_graph_plan.tensor_to_exec_tensor_mapping) {
    if (kv.second->producer()->num_outputs() == 0) {
      // 说明该tensor只是extra linker而并不会具有shape
      // 比如GroupOp
      continue;
    }
    auto it = shape_plan.find(kv.first);
    HT_ASSERT(it != shape_plan.end())
      << "can't find shape of tensor " << kv.second << " in the shape plan";
    exec_shape_plan[kv.second->id()] = it->second;
  }
  // exec graph中还有一些新增的tensor
  // 需要再进行一次额外的推导
  for (const auto& exec_tensor : exec_graph_plan.exec_graph->_record_exec_tensors) {
    auto& exec_op = exec_tensor->producer();
    HTShapeList exec_input_shapes;
    exec_input_shapes.reserve(exec_op->num_inputs());
    for (const auto& exec_input : exec_op->inputs()) {
      auto it = exec_shape_plan.find(exec_input->id());
      HT_ASSERT(it != exec_shape_plan.end()) 
        << "Something wrong, can't find the input shape of " << exec_input
        << " from the current exec shape plan!";
      exec_input_shapes.push_back(it->second);
    }
    HTShapeList exec_output_shapes = exec_op->InferShape(exec_input_shapes, runtime_ctx);
    auto exec_output_shapes_size = exec_output_shapes.size();
    for (size_t i = 0; i < exec_output_shapes_size; i++) {
      if (exec_op->output(i)->symbolic()) {
        if (is_SyShape_leaf(exec_op->output(i)->symbolic_shape())) {
          exec_op->output(i)->set_symbolic_shape(exec_output_shapes[i]);
        }
      }
      exec_shape_plan.insert(std::make_pair(exec_op->output(i)->id(), std::move(exec_output_shapes[i]))); // move constructor
    }
  }
  exec_graph_plan.shape_plan_pool.emplace_back(std::move(shape_plan));
  exec_graph_plan.exec_graph->AddShapePlan(std::move(exec_shape_plan));
}

// Should call in the order of global topo sort
DeviceGroupUnion DefineAndRunGraph::DeducePlacementGroup(Operator& op, Op2DGUnionMap& dg_union_map) {
  // 通过op meta指定了device group的
  if (op->device_group_hierarchy().size() > 0) {
    HT_ASSERT(!is_comm_op(op))
      << "Comm op shouldn't be provided with device group hierarchy to avoid chaos";
    dg_union_map[op->id()] = op->device_group_union();
    return op->device_group_union();
  } 
  // 未指定而需要推导的
  else {
    DeviceGroupUnion inferred;
    // 最后的group op特殊处理
    if (is_group_op(op)) {
      std::set<Device> devices;
      for (auto& input : op->in_dep_linkers()) {
        auto it = dg_union_map.find(input->producer()->id());
        HT_ASSERT(it != dg_union_map.end())
          << input->producer() << " should have deduced placement group before";
        auto dg_all = it->second.all();
        for (auto& device : dg_all.devices()) {
          devices.insert(device);
        }
      }
      inferred = DeviceGroupUnion({DeviceGroup(std::vector<Device>(devices.begin(), devices.end()))});
    } 
    // 其余的算子
    else {
      HT_ASSERT(op->num_inputs() > 0)
        << "Currently we cannot infer the devices "
        << "for operators with zero in-degree. : " << op;
      // 反向算子
      if (op->fw_op_id() != -1) {
        auto& fw_op = _op_indexing[op->fw_op_id()]; 
        // HT_LOG_WARN << op << " has fw op " << fw_op;
        // comm op要特殊处理
        // 这是因为comm op是唯一一个tensor和op的placement group不一样的算子
        // op group = src tensor group merge dst tensor group
        if (is_comm_op(fw_op)) {
          auto it = dg_union_map.find(fw_op->input(0)->producer()->id());
          HT_ASSERT(it != dg_union_map.end())
            << fw_op->input(0)->producer() << " should have deduced placement group before";
          inferred = it->second;
        } 
        // 与前向算子所在的placement group一致
        else {
          auto it = dg_union_map.find(fw_op->id());
          HT_ASSERT(it != dg_union_map.end())
            << fw_op << " should have deduced placement group before";
          inferred = it->second;
        }
      }
      // 前向算子或中间临时插入的辅助算子
      else {
        if (is_comm_op(op)) {
          auto& comm_op_impl = dynamic_cast<CommOpImpl&>(op->body());
          // 如果comm op没有提供dst group
          // 那么默认其和src group一样
          if (comm_op_impl.dst_group_hierarchy().size() == 0) {
            auto it = dg_union_map.find(op->input(0)->producer()->id());
            HT_ASSERT(it != dg_union_map.end())
              << op->input(0)->producer() << " should have deduced placement group before";
            inferred = it->second;   
          } 
          // 否则直接使用comm op提供的dst group
          else {
            inferred = comm_op_impl.dst_group_hierarchy().get(CUR_STRATEGY_ID);
          }
        }
        // 绝大多数情况走这条分支
        else {
          // 默认使用第一个输入所在的group 
          /*
          // is it a proper assumption?
          // i.e. attn_weights = ht.where(causal_mask, attn_weights, mask)
          // while causal_mask is on g0 but attn_weights is expected to be on g1
          auto it = dg_union_map.find(op->input(0)->producer()->id());
          HT_ASSERT(it != dg_union_map.end())
            << op->input(0)->producer() << " should have deduced placement group before";
          inferred = it->second; 
          */ 
          // 目前所有comm op均手动插入
          // 因此这里要保证所有placement group都一样
          for (auto& input : op->inputs()) {
            auto it = dg_union_map.find(input->producer()->id());
            HT_ASSERT(it != dg_union_map.end())
              << input->producer() << " should have deduced placement group before";
            if (inferred.size() == 0) {
              inferred = it->second;
            }
            HT_ASSERT(inferred.check_equal(it->second))
              << "Get different placement group unions"
              << ", the input " << input << " is " << it->second
              << ", but the history one is " << inferred;
          }
        }      
      }
    }
    dg_union_map[op->id()] = inferred;
    return inferred;
  }
}

void DefineAndRunGraph::Instantiate(OpRefList&& global_topo,
                                    Tensor2ShapeMap&& shape_plan,
                                    int32_t pipeline_num) {

  // deprecated: Test Case - 手动切换并行方案（验证切换时间）
  char* env = std::getenv("HETU_PARALLEL_CHANGE_TEST");
  if (env != nullptr) {
    if (std::string(env) == "COST" && change_parallel_test_case >= 1) {
      InstantiateTestCase(global_topo, shape_plan);
      change_parallel_test_case += 1;
      return;
    }
  }

  // initializations of the exec plan
  auto exec_graph_num = _exec_graph_plan_pool.size();
  Tensor2ShapeMap exec_shape_plan;
  Op2OpMap op_to_exec_op_mapping;
  Op2DGUnionMap op_to_pg_union_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  auto origin_param_buffer = std::make_shared<ParamBuffer>("origin_param_buffer");
  auto transfer_param_buffer = std::make_shared<ParamBuffer>("transfer_param_buffer");
  auto origin_param_and_optimizer_buffer = std::make_shared<ParamBuffer>("origin_param_and_optimizer_buffer");
  auto origin_param_and_optimizer_buckets = std::make_shared<ParamBuckets>("origin_param_and_optimizer_buckets");
  auto current_grad_buffer = std::make_shared<ParamBuffer>("current_grad_buffer");
  auto accumulate_grad_buffer = std::make_shared<ParamBuffer>("accumulate_grad_buffer");
  Tensor2TensorMap transfer_map;
  Tensor2TensorMap grad_map;
    
  exec_shape_plan.reserve(shape_plan.size());
  op_to_exec_op_mapping.reserve(_init_capacity);
  op_to_pg_union_mapping.reserve(_init_capacity);
  tensor_to_exec_tensor_mapping.reserve(_init_capacity);

  // initializations of the exec graph
  auto local_device = hetu::impl::comm::GetLocalDevice();
  auto exec_graph = Graph::_make_new_graph<ExecutableGraph>(name() + "_executable_" + std::to_string(exec_graph_num));
  exec_graph->NUM_STRATEGY = NUM_STRATEGY;
  exec_graph->CUR_STRATEGY_ID = CUR_STRATEGY_ID;
  // HT_LOG_INFO << local_device << ": instantiate " << exec_graph->name();
  Graph::push_graph_ctx(exec_graph->id());

  // assign pp stages
  // HT_LOG_WARN << local_device << ": Deduce pipeline";
  if (_multi_pipeline_maps.find(CUR_STRATEGY_ID) == _multi_pipeline_maps.end()) {
    _multi_pipeline_maps[CUR_STRATEGY_ID] = Device2PipelineMap();
    DeducePipeline(CUR_STRATEGY_ID, pipeline_num);
  }
  exec_graph->SetPipeline(_multi_pipeline_maps[CUR_STRATEGY_ID]);
  std::vector<int> used_ranks;
  for (const auto& kv : _multi_pipeline_maps[CUR_STRATEGY_ID]) {
    for (const auto& stage : kv.second) {
      for (const auto& device : stage.devices()) {
        auto rank = hetu::impl::comm::DeviceToWorldRank(device);
        if (std::find(used_ranks.begin(), used_ranks.end(), rank) == used_ranks.end()) {
          used_ranks.push_back(rank);
        }
      }
    }
  }
  std::sort(used_ranks.begin(), used_ranks.end());
  // HT_LOG_WARN << "used ranks = " << used_ranks;
  exec_graph->SetUsedRanks(used_ranks);
  exec_graph->SUGGESTED_HETERO_ID = SUGGESTED_HETERO_ID;
  SUGGESTED_HETERO_ID = 0;

  auto get_exec_input = [&](const Tensor& input) -> Tensor {
    auto it = tensor_to_exec_tensor_mapping.find(input->id());
    HT_RUNTIME_ERROR_IF(it == tensor_to_exec_tensor_mapping.end())
      << "Cannot find the executable version of Tensor " << input;
    return it->second;
  };

  // just use ds_hierarchy which was deduced in define_and_run_graph
  // executable_graph needn't deduce states again!
  auto handle_exec_output = [&](Tensor& tensor, Tensor& exec_tensor) -> void {
    HT_LOG_TRACE << "handle mapping of tensor " << tensor->id() << " " << tensor;
    // 1)、assign tensor mapping
    tensor_to_exec_tensor_mapping[tensor->id()] = exec_tensor;
    // 2)、assign shape
    auto plan_it = shape_plan.find(tensor->id());
    // The shape plan will be expanded step by step
    if (plan_it != shape_plan.end()) {
      // *only feed dict will set_shape
      exec_tensor->set_shape(plan_it->second);
      // HT_LOG_INFO << "set shape " << plan_it->second << " to " << exec_tensor;
    } else {
      // other shapes will be fixed and just recorded
      exec_tensor->set_shape(exec_tensor->shape()); // Workaround for Contiguous
      shape_plan[tensor->id()] = exec_tensor->shape();
    }
    exec_shape_plan[exec_tensor->id()] = exec_tensor->shape();
    HT_LOG_TRACE << "assign exec tensor " << exec_tensor << " shape " << exec_tensor->shape();
    exec_tensor->set_is_grad(tensor->is_grad());
    // 3)、assign symbolic shape
    // here symbolic shape will only used in some op
    // such as slice & reshape
    // the tensor shape is fixed and recorded in the shape_plan
    // note that no tensor will have an unknown shape in the exec graph
    if (tensor->symbolic()) {
      exec_tensor->copy_symbolic_shape(tensor->symbolic_shape());
      if (is_SyShape_leaf(exec_tensor->symbolic_shape())) {
        exec_tensor->set_symbolic_shape(exec_tensor->shape());
      }
    }
    // 4)、assign ds_hierarchy
    // just copy it from define graph
    exec_tensor->set_ds_hierarchy(tensor->ds_hierarchy());
    // HT_LOG_WARN << exec_tensor << " ds " << exec_tensor->cur_ds_union().ds_union_info();
    // 5)、assign add on inits
    auto it = _add_on_inits.find(tensor->id());
    if (_run_level != RunLevel::TOPO && it != _add_on_inits.end()) {
      Graph::ResetVariableData(exec_tensor, *it->second);
      // 考虑要切换plan，仅第一次使用_add_on_inits
      // 之后会使用热切换
      _add_on_inits.erase(tensor->id());
    }
    // 6)、assign param & optimizer states
    // 目前只是记录而并不会alloc
    if (_parameter_ops.find(tensor->producer()->id()) != _parameter_ops.end()
        && exec_tensor->producer()->placement_group_union().has(local_device)) {
      origin_param_buffer->AddTensor(exec_tensor);
    }
    if ((_parameter_ops.find(tensor->producer()->id()) != _parameter_ops.end()
        || _optimizer_variable_ops.find(tensor->producer()->id()) != _optimizer_variable_ops.end())
        && exec_tensor->producer()->placement_group_union().has(local_device)) {
      origin_param_and_optimizer_buffer->AddTensor(exec_tensor); // deprecates
      origin_param_and_optimizer_buckets->AddTensor(exec_tensor);
    }
  };

  HT_LOG_DEBUG << "Instantiating a " << type() << " graph with global topo " << global_topo;
  for (auto& op_ref : global_topo) {
    auto& op = op_ref.get();
    HT_LOG_TRACE << "Creating an executable version of op " << op << " begin...";

    // 前处理
    // 1、获取exec op的inputs
    // 2、推导placement group union
    // 3、进行autocast
    TensorList exec_inputs, exec_in_deps;
    std::tie(exec_inputs, exec_in_deps) = Operator::transform_each_input_tensor(op, get_exec_input);

    // symbolic shape debug use
    /*
    HTShapeList exec_input_shapes;
    for (auto& exec_input : exec_inputs) {
      exec_input_shapes.push_back(exec_input->shape());
    }
    HT_LOG_INFO << "Exec op " << op << " with inputs " << exec_inputs << " and shapes " << exec_input_shapes;
    */

    // HT_LOG_WARN << local_device << ": deduce placement group union for " << op;
    auto pg_union = DeducePlacementGroup(op, op_to_pg_union_mapping);
    // HT_LOG_WARN << local_device << ": placement group union for " << op << " is " << pg_union;

    auto autocast_id = AutoCast::cur_autocast_ctx();
    if (autocast_id != UINT64_MAX) {
      auto autocast = AutoCast::GetAutoCast(autocast_id);
      if (autocast.enabled()) {
        DataType datatype = DataType::UNDETERMINED;
        if (autocast.cast_type() != DataType::UNDETERMINED)
          datatype = autocast.cast_type();
        if (datatype != DataType::UNDETERMINED) {
          auto optype = op->type();
          if (is_optimizer_update_op(op) || is_host_to_device_op(op) || is_device_to_host_op(op) || is_data_transfer_op(op)) {
            // seems nothing to do
          } else {
            for (int i = 0; i < exec_inputs.size(); ++i) {
              if ((is_variable_op(exec_inputs[i]->producer()) || is_placeholder_op(exec_inputs[i]->producer())) &&
                  exec_inputs[i]->dtype() != datatype && 
                  (exec_inputs[i]->dtype() == DataType::BFLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT16 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT32 ||
                  exec_inputs[i]->dtype() == DataType::FLOAT64)) {
                if (transfer_map.find(exec_inputs[i]->id()) != transfer_map.end()) {
                  HT_LOG_TRACE << "Map " << &transfer_map << " reuse: " << exec_inputs[i]->id() << " -> " << transfer_map[exec_inputs[i]->id()]->id();
                  exec_inputs[i] = transfer_map[exec_inputs[i]->id()];
                } else {
                  auto& exec_op = Graph::MakeOp(std::make_shared<DataTransferOpImpl>(datatype, exec_inputs[i]->device()),
                                  {exec_inputs[i]}, OpMeta().set(exec_inputs[i]->producer()->op_meta()).set_name(exec_inputs[i]->producer()->name() + "_transfer").set_is_deduce_states(false), *exec_graph);
                  exec_op->MapToParallelDevices(exec_inputs[i]->placement_group_union());
                  HT_LOG_TRACE << "Map " << &transfer_map << " insert: " << exec_inputs[i]->id() << " -> " << exec_op->output(0)->id();
                  // we have to set the exec shape plan manually before the initialization of the plan
                  exec_shape_plan[exec_op->output(0)->id()] = exec_op->output(0)->shape();
                  exec_graph->_record_exec_tensors.emplace_back(exec_op->output(0));
                  exec_op->output(0)->set_ds_hierarchy(op->input(i)->ds_hierarchy()); // walkaround: set here by hand
                  if (_parameter_ops.find(op->input(i)->producer()->id()) != _parameter_ops.end()
                      && exec_inputs[i]->producer()->placement_group_union().has(local_device)) {
                    transfer_param_buffer->AddTensor(exec_op->output(0));
                  }
                  transfer_map[exec_inputs[i]->id()] = exec_op->output(0);
                  exec_inputs[i] = exec_op->output(0);
                }
              }
            }
          }
        }
      }
    }

    // 核心部分
    // only deduce ds hierarchy for define_and_run_graph, and copy directly for executable_graph
    // 注意MakeCommOp在InferMeta时不得不特殊处理
    // 需要从外面把CUR_HETERO_ID传进去
    if (is_comm_op(op)) {
      if (pg_union.has(local_device)) {
        exec_graph->CUR_HETERO_ID = pg_union.get_index(local_device);
      } else if (exec_inputs.at(0)->producer()->placement_group_union().has(local_device)) {
        exec_graph->CUR_HETERO_ID = exec_inputs.at(0)->producer()->placement_group_union().get_index(local_device);
      } else {
        exec_graph->CUR_HETERO_ID = exec_graph->SUGGESTED_HETERO_ID;
      }
    }
    // Debug use
    /*
    std::vector<HTShape> exec_input_shapes;
    for (const auto& exec_input : exec_inputs) {
      exec_input_shapes.emplace_back(exec_input->shape());
    }
    HT_LOG_WARN << local_device << ": make exec op for " << op
      << ", exec inputs are " << exec_inputs
      << ", exec input shapes are " << exec_input_shapes;
    */
    auto& exec_op = Graph::MakeOp(
      op->_body, std::move(exec_inputs),
      OpMeta().set(op->op_meta()).set_is_deduce_states(false).set_extra_deps(std::move(exec_in_deps)),
      *exec_graph);
    if (is_comm_op(op)) {
      /*
      HT_LOG_WARN << exec_op << " output shape is " << exec_op->output(0)->shape()
        << " input shape is " << exec_op->input(0)->shape()
        << " CUR_HETERO_ID is " << CUR_HETERO_ID;
      */
      exec_graph->CUR_HETERO_ID = 0;
    }

    // 后处理
    // 1、建立op和exec_op的映射
    // 2、给op和输出的tensor分配placement group union
    // 3、设置tensor的shape和ds_hierarchy
    // 4、标记param/optvar并给即将创建的exec graph预先设置ParamBuffer
    // 5、给grad设置placement和buffer
    op_to_exec_op_mapping[op->id()] = exec_op;
    exec_op->MapToParallelDevices(pg_union);
    /*
    if (is_comm_op(exec_op))
      HT_LOG_WARN << exec_op << " placement group is " << pg_union;
    */
    Operator::for_each_output_tensor_pair(op, exec_op, handle_exec_output);
    if (_parameter_ops.find(op->id()) != _parameter_ops.end()) {
      Graph::MarkAsParameter(exec_op);
    }
    if (_optimizer_variable_ops.find(op->id()) != _optimizer_variable_ops.end()) {
      Graph::MarkAsOptimizerVariable(exec_op);
    }
    if (is_optimizer_update_op(exec_op)) {
      Tensor& param = op->input(0);
      Tensor& exec_param = exec_op->input(0);
      Tensor& exec_grad = exec_op->input(1);
      HT_ASSERT(exec_graph->_parameter_ops.find(exec_param->producer()->id()) != exec_graph->_parameter_ops.end())
        << "optimizer op " << exec_op << " input 0 " << exec_param << " is not a parameter";
      // zero属性已经类似multi_ds一样设置成了list
      /*
      auto zero = (param->get_distributed_states().get_dim(-1) > 1) && param->get_distributed_states().zero();
      auto adam_op_interface = std::dynamic_pointer_cast<AdamOpImpl>(exec_op->_body);
      if (adam_op_interface) {
        adam_op_interface->set_zero(zero);
      }
      */
      // 热切换接口需要提前设置一些grad的信息
      exec_grad->producer()->set_device_group_hierarchy(exec_param->producer()->device_group_hierarchy());
      if (exec_grad->producer()->placement_group_union().has(local_device)) {
        current_grad_buffer->AddTensor(exec_grad);
        accumulate_grad_buffer->AddTensor(exec_grad);
        exec_grad->set_placement(local_device);
        HT_LOG_TRACE << "local grad " << exec_grad << " ds union = " << exec_grad->cur_ds_union().ds_union_info();
      }
      grad_map[exec_param->id()] = exec_grad;
    }
    HT_LOG_TRACE << "Creating an executable version of op " << op << " end...";
  }

  // assign fw_op_id map
  for (auto& op_ref : global_topo) {
    auto& op = op_ref.get();
    auto& exec_op = op_to_exec_op_mapping[op->id()];
    if (op->fw_op_id() != -1) {
      exec_op->set_fw_op_id(op_to_exec_op_mapping[op->fw_op_id()]->id());
    } 
  }
  
  // assign initial shape plan
  exec_graph->AddShapePlan(std::move(exec_shape_plan));

  // assign param buffer, grad buffer and transfer map
  exec_graph->_origin_param_buffer = std::move(origin_param_buffer);
  exec_graph->_transfer_param_buffer = std::move(transfer_param_buffer);
  exec_graph->_origin_param_and_optimizer_buffer = std::move(origin_param_and_optimizer_buffer);
  exec_graph->_origin_param_and_optimizer_buckets = std::move(origin_param_and_optimizer_buckets);
  exec_graph->_current_grad_buffer = std::move(current_grad_buffer);
  exec_graph->_accumulate_grad_buffer = std::move(accumulate_grad_buffer);
  exec_graph->_transfer_map = std::move(transfer_map);
  exec_graph->_grad_map = std::move(grad_map);
  
  // wrap up all of this as an exec graph plan
  _exec_graph_plan_pool.emplace_back(std::move(exec_graph), 
                                     std::move(op_to_exec_op_mapping),
                                     std::move(tensor_to_exec_tensor_mapping),
                                     std::move(global_topo),
                                     std::vector<Tensor2ShapeMap>{std::move(shape_plan)},
                                     CUR_STRATEGY_ID);

  Graph::pop_graph_ctx();
  // HT_LOG_WARN << "Instantiating end";

  // deprecated: Test Case - 手动切换并行方案
  if (env != nullptr) {
    if (std::string(env) == "PRECISION" || std::string(env) == "COST") {
      change_parallel_test_case += 1;
    }
  }
}

// 每次调用run都会从当前的define graph中
// 生成/使用之前生成过的一个exec graph
// 而只有当：
// 1、并行策略 2、fetch的tensor 
// 与cache的某一个重合时，才会复用
// 目前的写法下，我们认为并行策略已经在python端选择好了然后再传进来
// 2024.3.6 update:
// 目前一个exec graph支持多个shape plan
// 即允许feed_dict的shape（包括batch_size以及seq_len等）可变
NDArrayList DefineAndRunGraph::Run(const Tensor& loss, const TensorList& fetches,
                                   const FeedDict& feed_dict, const int num_micro_batches,
                                   const int cur_strategy_id, RunLevel run_level, const double grad_scale) {
  _run_level = run_level;
  CUR_STRATEGY_ID = static_cast<size_t>(cur_strategy_id);
  auto local_device = hetu::impl::comm::GetLocalDevice(); // only for debug use
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph begin...";

  // get feed dict shape
  Tensor2ShapeMap feed_dict_shape;
  for (const auto& kv : feed_dict) {
    if (!kv.second.is_defined()) 
      continue; 
    // TODO: use NDArrayMeta::split instead, but currently no support for arg chunk_num
    auto micro_batches = NDArray::split(kv.second, num_micro_batches);
    // currently assume that all micro batches have the same shape
    feed_dict_shape[kv.first] = micro_batches[0]->shape();
  }

  size_t next_active_exec_plan;
  size_t next_active_shape_plan;
  size_t exec_plan_pool_size = _exec_graph_plan_pool.size();
  bool in_exec_plan_pool = false;
  for (size_t i = 0; i < exec_plan_pool_size; i++)  {
    const auto& exec_graph_plan = _exec_graph_plan_pool[i];
    bool exec_plan_matched = true;
    // 先看strategy匹配不
    if (static_cast<size_t>(cur_strategy_id) != exec_graph_plan.strategy_id) {
      exec_plan_matched = false;
    }
    // 再看fetch匹配不
    for (const auto& fetch : fetches) {
      if (std::find(exec_graph_plan.fetches.begin(), exec_graph_plan.fetches.end(), fetch) == exec_graph_plan.fetches.end()) {
        HT_LOG_TRACE << local_device << ": exec_graph_plan fetches are " << exec_graph_plan.fetches 
          << " and the mismatch fetch is " << fetch;
        exec_plan_matched = false;
        break;
      }
    }
    if (exec_plan_matched) {
      HT_LOG_TRACE << local_device << ": plan matched";
      in_exec_plan_pool = true;
      next_active_exec_plan = i;
      break;
    }
  }

  // 需要创建一个新的exec graph
  // 用当前feed dict的shape先初始化一套shape plan
  // 作为该exec graph的shape plan pool里的第一个
  if (!in_exec_plan_pool) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new exec graph to the pool begin...";
    Tensor2ShapeMap shape_plan;
    // 后续会由feed_dict的shape在MakeOp时推导出所有的shape
    for (const auto& kv : feed_dict) {
      shape_plan[kv.first] = feed_dict_shape[kv.first];
    }
    auto is_feed_dict_op = [&](const Operator& op) -> bool {
      return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
        return feed_dict.find(tensor->id()) != feed_dict.end();
      });
    };
    OpRefList global_topo = Graph::TopoSort(fetches, -1, is_feed_dict_op);
    HT_LOG_DEBUG << local_device << ": global topo of define graph is " << global_topo;
    // Instantiate会将新的exec_graph_plan加入pool中
    int32_t pipeline_num = 0;
    if (!loss->cur_ds_union().is_hetero()) {
      pipeline_num = loss->cur_ds_union().get(0).states(0);
    } else if (loss->cur_ds_union().hetero_dim() == 0) {
      pipeline_num = loss->cur_ds_union().size();
    } else {
      HT_RUNTIME_ERROR << "Currently we use the ds of loss to deduce pipeline num"
        << ", so the ds union of loss shouldn't be hetero on other dim except for 0";
    }
    Instantiate(std::move(global_topo), std::move(shape_plan), pipeline_num);
    // 补上fetches（其在instantiate中不需要用到，但是plan需要进行记录）
    auto& new_plan = _exec_graph_plan_pool.back();
    new_plan.fetches = fetches;
    // 新的exec plan就是exec plan pool中的最后一个
    next_active_exec_plan = _exec_graph_plan_pool.size() - 1;
    // 新的shape plan就是shape plan pool中的第一个
    next_active_shape_plan = 0; 
    HT_LOG_DEBUG << local_device << ": [Graph Plan] add a new shape plan and an exec graph to the pool end...";
  } 
  // 命中pool中已有的exec graph
  // 但可能feed dict不一样
  // 这种情况下我们不需要生成新的exec graph
  // 但需要推导新的shape plan
  else {
    auto& exec_graph_plan = _exec_graph_plan_pool[next_active_exec_plan];
    auto shape_plan_pool_size = exec_graph_plan.shape_plan_pool.size();
    bool in_shape_plan_pool = false;
    for (size_t i = 0; i < shape_plan_pool_size; i++) {
      const auto& shape_plan = exec_graph_plan.shape_plan_pool[i];
      bool shape_plan_matched = true;
      for (const auto& kv : feed_dict) {
        if (!kv.second.is_defined()) continue;
        auto it = shape_plan.find(kv.first);
        // 1、有可能是feed_dict发生了改变（在依据global topo生成的shape plan中没有feed dict）
        // 2、有可能是feed_dict的shape发生了改变（shape对不上）
        HT_LOG_TRACE << local_device << ": shape plan is " << shape_plan << " and key to match is "
          << kv.first << ":" << feed_dict_shape[kv.first];
        if (it == shape_plan.end() || it->second != feed_dict_shape[kv.first]) {
          shape_plan_matched = false;
          break;
        }
      }
      if (shape_plan_matched) {
        in_shape_plan_pool = true;
        next_active_shape_plan = i;
        break;
      }
    }
    // 如果不在shape_plan_pool中
    // 需要推导新的shape plan
    if (!in_shape_plan_pool) {
      DeduceShapePlan(exec_graph_plan, feed_dict, feed_dict_shape);                                            
      // 新的shape plan就是shape plan pool中的最后一个
      next_active_shape_plan = exec_graph_plan.shape_plan_pool.size() - 1;
    }
  }

  // 需要切换exec graph
  if (!_is_active || _active_exec_plan != next_active_exec_plan) {
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan begin...";
    // 热切换
    if (_is_active) {
      auto key = std::make_pair(_active_exec_plan, next_active_exec_plan);
      if (_param_switcher_pool.find(key) == _param_switcher_pool.end()) {
        _param_switcher_pool[key] = std::make_shared<SwitchExecGraph>(this, _active_exec_plan, next_active_exec_plan);
        _grad_switcher_pool[key] = std::make_shared<SwitchExecGraph>(this, _active_exec_plan, next_active_exec_plan);
      }
      // 旧的exec graph
      auto& old_exec_graph = _exec_graph_plan_pool[_active_exec_plan].exec_graph;
      auto& new_exec_graph = _exec_graph_plan_pool[next_active_exec_plan].exec_graph;
      // 默认的切换状态设置
      auto param_switch_mode = SWITCH_MODE::SWITCH_TRANSFER_PARAM;
      auto grad_switch_mode = SWITCH_MODE::SWITCH_ACCUMULATE_GRAD;
      auto param_switch_level = SWITCH_LEVEL::EXEC;
      auto grad_switch_level = SWITCH_LEVEL::EXEC;
      // 1、----- level设置 -----
      // 1)、topo前只能跟topo（算exec graph和switcher的topo）
      // 2)、alloc前只能跟topo或alloc或update（新的一轮开始）
      // 3)、grad后只能跟grad或update（grad要么不断累积要么更新掉）
      // 4)、update前都能跟
      // 其实原则就是有transfer param就切，没有就切origin param
      // 有accumulate grad就切，没有就不切
      if (_run_level == RunLevel::TOPO) {
        HT_ASSERT(old_exec_graph->_run_level == RunLevel::TOPO) 
          << "graph with RunLevel::TOPO should only follow behind graph with RunLevel::TOPO right now";
      }
      if (_run_level == RunLevel::ALLOC) {
        HT_ASSERT(old_exec_graph->_run_level == RunLevel::TOPO
                  || old_exec_graph->_run_level == RunLevel::ALLOC
                  || old_exec_graph->_run_level == RunLevel::UPDATE) 
          << "graph with RunLevel::ALLOC should only follow behind graph with RunLevel::TOPO or RunLevel::ALLOC or RunLevel::UPDATE right now";
      }
      if (old_exec_graph->_run_level == RunLevel::GRAD) {
        HT_ASSERT(_run_level == RunLevel::GRAD
                  || _run_level == RunLevel::UPDATE) 
          << "graph with RunLevel::GRAD should only followed by graph with RunLevel::GRAD or RunLevel::UPDATE right now";
      }
      // 如果旧的exec graph只是建立topo
      // 其并没有产生param和grad
      if (old_exec_graph->_run_level == RunLevel::TOPO) {
        param_switch_level = SWITCH_LEVEL::TOPO;
        grad_switch_level = SWITCH_LEVEL::TOPO;
      }
      // 如果旧的exec graph只是alloc
      // 其并没有产生grad
      if (old_exec_graph->_run_level == RunLevel::ALLOC) {
        grad_switch_level = SWITCH_LEVEL::TOPO;
      }
      // 如果旧的exec graph是update
      // grad已经被消耗掉了
      if (old_exec_graph->_run_level == RunLevel::UPDATE) {
        grad_switch_level = SWITCH_LEVEL::TOPO;
      }
      // 2、----- mode设置 -----
      // 如果旧的exec graph没开AMP
      // 或者是刚刚进行了update（使得transfer param是空的）
      // 那么只能切换origin param buffer
      // 2024.5.20 Update: 
      // 将optimzer和origin param放到一个buffer中
      // 但目前在hetero+zero情形下无法使用
      if (old_exec_graph->_transfer_param_buffer->IsEmpty()
          || (old_exec_graph->_run_level == RunLevel::UPDATE
              && param_switch_level == SWITCH_LEVEL::EXEC)) {
        if (old_exec_graph->_use_origin_param_and_optimizer_buffer
            || old_exec_graph->_use_origin_param_and_optimizer_buckets) {
          param_switch_mode = SWITCH_MODE::SWITCH_ORIGIN_PARAM_AND_OPTIMIZER;
        } else {
          param_switch_mode = SWITCH_MODE::SWITCH_ORIGIN_PARAM;
        }
      }
      // 3、----- buffer释放 -----
      // 如果旧的exec graph是grad
      // 那么热切换需要释放之前的current grad buffer
      // 如果旧的exec graph是update
      // 那么热切换需要释放之前的transfer param buffer和current grad buffer
      if (old_exec_graph->_run_level == RunLevel::GRAD) {
        if (old_exec_graph->_use_current_grad_buffer) {
          if (!old_exec_graph->_current_grad_buffer->IsEmpty()) {
            HT_ASSERT(old_exec_graph->_current_grad_buffer->IsAllocated())
              << "old exec graph with RunLevel::GRAD should have allocated the current grad buffer";
            old_exec_graph->_current_grad_buffer->Free();
          }
        }
      }
      if (old_exec_graph->_run_level == RunLevel::UPDATE) {
        if (!old_exec_graph->_transfer_param_buffer->IsEmpty()) {
          HT_ASSERT(old_exec_graph->_transfer_param_buffer->IsAllocated())
            << "old exec graph with RunLevel::UPDATE should have allocated the transfer param buffer";
          old_exec_graph->_transfer_param_buffer->Free();
        }
        if (old_exec_graph->_use_current_grad_buffer) {
          if (!old_exec_graph->_current_grad_buffer->IsEmpty()) {
            HT_ASSERT(old_exec_graph->_current_grad_buffer->IsAllocated())
              << "old exec graph with RunLevel::UPDATE should have allocated the current grad buffer";
            old_exec_graph->_current_grad_buffer->Free();
          }
        }
        // 显存池当快要OOM时会自动处理
        // 但目前发现那样会很慢
        // 因此这里手动清空
        hetu::impl::ProfileAfterEmptyAllCUDACache(local_device);
        hetu::impl::comm::EmptyNCCLCache();
        // GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " after empty cache");
      }
      /*
      for (auto& tensor : _exec_graph_plan_pool[next_active_exec_plan].exec_graph->_transfer_param_buffer->tensor_list()) {
        HT_LOG_INFO << local_device << ": transfer param " << tensor << " meta is " << tensor->meta() << " and device group is " << tensor->producer()->device_group()
          << " and ds is: " << tensor->get_distributed_states().ds_info();
      }
      for (auto& tensor : _exec_graph_plan_pool[next_active_exec_plan].exec_graph->_accumulate_grad_buffer->tensor_list()) {
        HT_LOG_INFO << local_device << ": accumulate grad " << tensor << " meta is " << tensor->meta() << " and device group is " << tensor->producer()->device_group()
          << " and ds is: " << tensor->get_distributed_states().ds_info();
      }
      */
      // 实际热切换
      // 目前已修改成async版本
      // 如果要改成非async的
      // 更改环境变量HETU_SWITCH_PROFILE低于TIME即可
      // TODO: 实现CPU上的Switch（例如AdamOp的step，其目前并不在buffer中）
      if (param_switch_mode != SWITCH_MODE::SWITCH_ORIGIN_PARAM_AND_OPTIMIZER
          || old_exec_graph->_use_origin_param_and_optimizer_buckets == false) {
        _param_switcher_pool[key]->SwitchParams(param_switch_mode, param_switch_level, "switch params and opt-states");
      }
      // 按buckets的顺序进行switch
      else {
        size_t buckets_size = old_exec_graph->_origin_param_and_optimizer_buckets->buckets_size();
        if (_param_and_opt_var_bucket_switcher_pool.find(key) == _param_and_opt_var_bucket_switcher_pool.end()) {
          // 统一使用全局的通信组
          // TODO: 后续使用实际参与的所有device
          std::unordered_set<Device> comm_set = {};
          const auto& global_device_group = hetu::impl::comm::GetGlobalDeviceGroup();
          for (const auto& device : global_device_group.devices()) {
            comm_set.emplace(device);
          }
          _param_and_opt_var_bucket_switcher_pool[key] = std::vector<std::shared_ptr<SwitchExecGraph>>();
          for (int32_t bucket_num = 0; bucket_num < buckets_size; bucket_num++) {
            _param_and_opt_var_bucket_switcher_pool[key].emplace_back(std::make_shared<SwitchExecGraph>(this, _active_exec_plan, next_active_exec_plan, bucket_num, comm_set));
          }
        }
        // tricky part
        // topo caculation could be "overlapped"
        for (int32_t bucket_num = 0; bucket_num < buckets_size; bucket_num++) {
          _param_and_opt_var_bucket_switcher_pool[key][bucket_num]->SwitchParams(param_switch_mode, SWITCH_LEVEL::TOPO, "switch params and opt-states bucket " + std::to_string(bucket_num));
        }
        auto& global_mpi_group = hetu::impl::comm::MPICommunicationGroup::GetOrCreateWorldwide();
        global_mpi_group->Barrier(true);
        TIK(switch_buckets_time);
        for (int32_t bucket_num = 0; bucket_num < buckets_size; bucket_num++) {
          _param_and_opt_var_bucket_switcher_pool[key][bucket_num]->SwitchParams(param_switch_mode, param_switch_level, "switch params and opt-states bucket " + std::to_string(bucket_num));
        }
        SynchronizeAllStreams();
        global_mpi_group->Barrier(true);
        TOK(switch_buckets_time);
        HT_LOG_WARN << "switch buckets time = " << COST_MSEC(switch_buckets_time) << " ms";
      }
      if (!(grad_switch_level == SWITCH_LEVEL::TOPO && !_need_grad_switch_topo)) {
        _grad_switcher_pool[key]->SwitchParams(grad_switch_mode, grad_switch_level, "switch grads");
      }
    }
    _is_active = true;
    _active_exec_plan = next_active_exec_plan;
    HT_LOG_DEBUG << local_device << ": [Graph Plan] Context switch to the new exec plan end...";
  }
  HT_LOG_DEBUG << local_device << ": [Graph Plan] obtain exec graph end...";

  // deprecated: Test Case - 手动切换并行方案（验证切换时间）
  char* env = std::getenv("HETU_PARALLEL_CHANGE_TEST");
  if (env != nullptr) {
    if (std::string(env) == "COST" && change_parallel_test_case >= 1) {
      return {};
    }
  }

  // 运行挑选出的active exec graph
  auto& exec_graph = _exec_graph_plan_pool[_active_exec_plan].exec_graph;
  auto& op_to_exec_op_mapping = _exec_graph_plan_pool[_active_exec_plan].op_to_exec_op_mapping;
  auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
  auto& exec_loss = tensor_to_exec_tensor_mapping[loss->id()]; 
  TensorList exec_fetches;
  FeedDict exec_feed_dict;

  // 设置shape plan
  HT_LOG_DEBUG << exec_graph->name() << " use shape plan " << next_active_shape_plan;
  exec_graph->SetShapePlan(next_active_shape_plan);

  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    HT_ASSERT(tensor_to_exec_tensor_mapping.find(fetch->id()) != tensor_to_exec_tensor_mapping.end())
      << "can't find fetch tensor " << fetch << " in the mapping";
    exec_fetches.push_back(tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict) {
    if (tensor_to_exec_tensor_mapping.find(kv.first) == tensor_to_exec_tensor_mapping.end()) {
      HT_LOG_DEBUG << "feed tensor " << kv.first << " is not used in the exec graph"
        << ", so we just skipped it";
      continue;
    }
    exec_feed_dict[tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  }
  // 验证mempool是否能释放干净
  // GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " before empty cache");
  // hetu::impl::ProfileAfterEmptyAllCUDACache(local_device);
  HT_LOG_DEBUG << exec_graph->name() << " start running..." ;
  NDArrayList ret;
  exec_graph->SetRunLevel(run_level);
  if (exec_graph->NeedRank(hetu::impl::comm::DeviceToWorldRank(local_device))) {
    Graph::push_graph_ctx(exec_graph->id()); // 防止exec graph run内部MakeOp时忘记加
    ret = exec_graph->Run(exec_loss, exec_fetches, exec_feed_dict, num_micro_batches, 
                          cur_strategy_id, run_level, grad_scale);
    Graph::pop_graph_ctx();
  }
  // 释放graph切换相关的event
  exec_graph->_switch_param_events.clear();
  exec_graph->_switch_grad_events.clear();
  // 验证mempool是否能释放干净
  // hetu::impl::ProfileAfterEmptyAllCUDACache(local_device);
  // GetCUDAProfiler(local_device)->PrintCurrMemoryInfo(name() + " after empty cache");
  return ret;
}

// TODO: merge two `Run` func
NDArrayList DefineAndRunGraph::Run(const TensorList& fetches,
                                   const FeedDict& feed_dict) {
  HT_RUNTIME_ERROR << "NotImplementedError";
  /*
  bool has_uninstantiated_ops =
    std::any_of(fetches.begin(), fetches.end(), [&](const Tensor& fetch) {
      return _op_to_exec_op_mapping.find(fetch->producer_id()) ==
        _op_to_exec_op_mapping.end();
    });
  if (has_uninstantiated_ops)
    Instantiate();
  TensorList exec_fetches;
  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    exec_fetches.push_back(_tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  FeedDict exec_feed_dict;
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict)
    exec_feed_dict[_tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  return _exec_graph->Run(exec_fetches, exec_feed_dict);
  */
}

} // namespace graph
} // namespace hetu
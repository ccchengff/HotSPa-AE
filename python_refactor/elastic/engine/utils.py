from typing import List

class Args:
    @staticmethod
    def print_args_list(args_list: List["Args"]):
        for item in args_list:
            print(item)
            
    def _type_check(self, value, *types):
        for cur_type in types:
            if isinstance(value, cur_type):
                return value
        raise TypeError(f"Value {value} must in types {types}.")
    
    def __str__(self):
        attrs = vars(self)
        attrs_str = ', '.join(f'{key} = {value}' for key, value in attrs.items())
        return f'{self.__class__.__name__}({attrs_str})'
    
class TrainerCtxs(Args):
    def __init__(self, **kwargs):
        self.bf16: bool = kwargs["bf16"]
        # 预先profile出来的
        self.hetero_tp_alpha: List[float] = kwargs["hetero_tp_alpha"]
        self.hetero_tp_weight: List[float] = kwargs["hetero_tp_weight"]
        self.normal_layers: int = kwargs["normal_layers"]
        self.normal_mbn: int = kwargs["normal_mbn"]
        self.normal_compute_time: int = kwargs["normal_compute_time"]
        self.memory_k: List[float] = kwargs["memory_k"]
        self.memory_embedding: float = kwargs["memory_embedding"]
        self.memory_extra: float = kwargs["memory_extra"]
        # bias细分为embedding和extra
        # embedding是只有第一个stage和最后一个stage具有的异构层
        # extra是各种cuda/nccl ctx等额外显存占用
        # self.memory_d: List[float] = kwargs["memory_d"]
        self.memory_bound: float = kwargs["memory_bound"]
        self.memory_safe_gap: float = kwargs["memory_safe_gap"]
        self.straggler_threshold: float = kwargs["straggler_threshold"]
        self.straggler_safe_gap: float = kwargs["straggler_safe_gap"]
        self.top_k: int = kwargs["top_k"]
        
class TrainerDatasetArgs(Args):
    def __init__(self, **kwargs):
        self.dataset = kwargs["dataset"]
        self.consumed_samples: int = kwargs["consumed_samples"]
        self.steps: int = kwargs["steps"]
        self.epochs: int = kwargs["epochs"]
        self.step: int = kwargs["step"]
        self.epoch: int = kwargs["epoch"]
        
class TrainerStrategyArgs(Args):
    def __init__(self, **kwargs):
        self.dp: int = kwargs["dp"]
        self.tp: int = kwargs["tp"]
        self.pp: int = kwargs["pp"]
        self.zero: bool = kwargs["zero"]
        self.rank_to_device_mapping: Dict[int, Any] = kwargs["rank_to_device_mapping"]
        self.suspended_rank_list: List[int] = kwargs["suspended_rank_list"]
        self.unused_rank_list: List[int] = kwargs["unused_rank_list"]
        self.hetero_data: bool = kwargs["hetero_data"]
        self.hetero_layers: List[List[int]] = kwargs["hetero_layers"]
        self.hetero_stages: List[int] = kwargs["hetero_stages"]
        self.hetero_micro_batch_num_list: List[int] = kwargs["hetero_micro_batch_num_list"]
        
class TrainerCommArgs(Args):
    def __init__(self, **kwargs):
        self.input_ds_union: ht.DistributedStatesUnion = kwargs["input_ds_union"]
        self.input_device_group_union: List[ht.DeviceGroup] = kwargs["input_device_group_union"]
        self.label_ds_union: ht.DistributedStatesUnion = kwargs["label_ds_union"]
        self.label_device_group_union: List[ht.DeviceGroup] = kwargs["label_device_group_union"]
        self.local_device: ht.device = kwargs["local_device"]
        self.all_devices: ht.DeviceGroup = kwargs["all_devices"]
        
class TrainerEnvs(Args):
    def __init__(self, **kwargs):
        self.run_straggler_experiment: bool = kwargs["run_straggler_experiment"]
        self.run_memory_experiment: bool = kwargs["run_memory_experiment"]
        self.straggler_file: str = kwargs["straggler_file"]
        self.memory_file: str = kwargs["memory_file"]
        self.elastic: bool = kwargs["elastic"]
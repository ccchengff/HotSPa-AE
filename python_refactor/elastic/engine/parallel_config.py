import json
import hetu
from queue import Queue

def generate_gpt_3d_config(rank_to_device_mapping, unused_rank, hetero_layers, hetero_stages, num_layers=32, num_gpus=8, dp=2, tp=2, pp=2, zero=True):
    
    accumulate_val = 0
    accumulate_hetero_stages = [0,]
    for val in hetero_stages:
        accumulate_val += val
        accumulate_hetero_stages.append(accumulate_val)
    
    if dp == 1:
        zero = False
    num_devices_per_stage = num_gpus // pp
    
    dp_union = [dp for _ in range(dp)]
    tp_union_list = []
    dg_union_list = []
    for block_id in range(num_layers):
        hybrid_tp_degree = []
        hybrid_device_group = []
        for pipeline_id in range(dp):
            device_group_num = 0
            cnt = 0
            for hetero_layer in hetero_layers[pipeline_id]:
                cnt += hetero_layer
                if block_id < cnt:
                    break
                device_group_num += 1
            ranks = range(device_group_num * tp + accumulate_hetero_stages[pipeline_id] * tp, 
                          (device_group_num + 1) * tp + accumulate_hetero_stages[pipeline_id] * tp)
            hybrid_tp_degree.append(len([rank for rank in ranks if rank not in unused_rank]))
            hybrid_device_group.append([rank_to_device_mapping[rank] for rank in ranks if rank not in unused_rank])
        tp_union_list.append(hybrid_tp_degree)
        dg_union_list.append(hybrid_device_group)

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
        'input': {
            'split': {'0': dp_union},
            'dup': tp_union_list[0],
            'device_group_union': dg_union_list[0],
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': tp_union_list[0]},
                'dup': dp_union,
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'wpe': {
                'split': {},
                'dup': [tp_union_list[0][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {},
                'dup': [tp_union_list[-1][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[-1],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': tp_union_list[-1]},
            'dup': dp_union,
            'device_group_union': dg_union_list[-1],
            'type': 'variable'
        },
        'label': {
            'split': {'0': dp_union},
            'dup': tp_union_list[-1],
            'device_group_union': dg_union_list[-1],
            'type': 'placeholder'
        }
    }
    
    for block_id in range(num_layers):
        blocks_json = ds_parallel_config['gpt']['blocks']
        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'layernorm1': {
                'split': {},
                'dup': [tp_union_list[block_id][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {},
                'dup': [tp_union_list[block_id][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            }
        }
    return ds_parallel_config

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    config_paths = args.ds_parallel_config.split(',')
    assert len(config_paths) == args.num_strategy, \
      f'ds_parallel_config num should equal to num_strategy {args.num_strategy}'
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = json.load(open(config_path, 'r'))
        ds_parallel_configs.append(config_spread_zero(ds_parallel_config))
    return ds_parallel_configs

def config_spread_zero(ds_parallel_config):
    zero = ds_parallel_config['zero']
    # assign zero to all variables
    config_queue = Queue()
    for value in ds_parallel_config.values():
        config_queue.put(value)
    while (not config_queue.empty()):
        config = config_queue.get()
        if type(config) == dict:
            if 'type' in config:
                if config['type'] == 'variable' and 'zero' not in config:
                    config['zero'] = zero
            else:
                for value in config.values():
                    config_queue.put(value)
    return ds_parallel_config

def get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    multi_ds_parallel_config = []
    for ds_parallel_config in ds_parallel_configs:
        config_queue = Queue()
        config_queue.put(ds_parallel_config)
        while (not config_queue.empty()):
            config = config_queue.get()
            if module_name in config:
                multi_ds_parallel_config.append(config[module_name])
                break
            else:
                for value in config.values():
                    if type(value) == dict:
                        if "range" in value and (_range < value["range"][0] or _range > value["range"][-1]):
                            continue
                        config_queue.put(value)
    assert len(multi_ds_parallel_config) == len(ds_parallel_configs), 'ds_parallel_configs parse error!'
    return multi_ds_parallel_config

def parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    ds_hierarchy = []
    dg_hierarchy = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds_union, dg_union = config2ds(ds_parallel_config)
        ds_hierarchy.append(ds_union)
        dg_hierarchy.append(dg_union)
    return ds_hierarchy, dg_hierarchy

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    ds_list = []
    dg_list = []
    if config['type'] == 'placeholder':
        hetero_dim = 0
    elif config['type'] == 'variable':
        hetero_dim = -1
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")   
    hetero_sum = len(config['device_group_union'])
    if hetero_sum == 1:
        hetero_dim = -3
    for hetero_num in range(hetero_sum):
        dummy_num_devices = len(config['device_group_union'][hetero_num]) * hetero_sum
        zero = False
        split = {}
        for key, value in config['split'].items():
            assert len(value) == hetero_sum, "hetero sum mismatches"
            split[int(key)] = value[hetero_num]
        assert len(config['dup']) == hetero_sum, "hetero sum mismatches"
        states = {-1: config['dup'][hetero_num], **split}
        if config['type'] == 'placeholder':
            order = sorted(split.keys()) + [-1]
        elif config['type'] == 'variable':
            order = [-1] + sorted(split.keys())
            assert 'zero' in config, f"variable config must have zero!"
            zero = config['zero']
        else:
            raise RuntimeError(f"unsupported type {config['type']}!")
        ds = hetu.DistributedStates(dummy_num_devices, states, order, zero)
        all_devices = hetu.global_device_group()
        dg = hetu.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group_union'][hetero_num]])
        ds_list.append(ds)
        dg_list.append(dg)
    return hetu.DistributedStatesUnion(ds_list, hetero_dim), dg_list
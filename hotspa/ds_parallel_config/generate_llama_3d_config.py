import argparse
import json
import os

def generate_llama_3d_config(num_layers=32, num_gpus=8, dp=2, tp=2, pp=2, zero=True):
    if dp == 1:
        zero = False
    num_layers_per_stage = num_layers // pp
    num_devices_per_stage = num_gpus // pp
    device_groups = [list(range(stage_id * num_devices_per_stage, (stage_id + 1) * num_devices_per_stage)) for stage_id in range(pp)]

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
        'input': {
            'split': {'0': [dp]},
            'dup': [tp],
            'device_group_union': [device_groups[0]],
            'type': 'placeholder'
        },
        'llama': {
            'wte': {
                'split': {'0': [tp]},
                'dup': [dp],
                'device_group_union': [device_groups[0]],
                'type': 'variable'
            },
            'wpe': {
                'split': {},
                'dup': [dp * tp],
                'device_group_union': [device_groups[0]],
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {},
                'dup': [dp * tp],
                'device_group_union': [device_groups[-1]],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': [tp]},
            'dup': [dp],
            'device_group_union': [device_groups[-1]],
            'type': 'variable'
        },
        'label': {
            'split': {'0': [dp]},
            'dup': [tp],
            'device_group_union': [device_groups[-1]],
            'type': 'placeholder'
        }
    }

    for stage_id in range(pp):
        block_start_id = num_layers_per_stage * stage_id
        block_end_id = num_layers_per_stage * (stage_id + 1) - 1
        blocks_json = ds_parallel_config['llama']['blocks']
        blocks_json[f'blocks{block_start_id}-{block_end_id}'] = {
            'range': [block_start_id, block_end_id],
            'layernorm1': {
                'split': {},
                'dup': [dp * tp],
                'device_group_union': [device_groups[stage_id]],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': [tp]},
                    'dup': [dp],
                    'device_group_union': [device_groups[stage_id]],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': [tp]},
                    'dup': [dp],
                    'device_group_union': [device_groups[stage_id]],
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {},
                'dup': [dp * tp],
                'device_group_union': [device_groups[stage_id]],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': [tp]},
                    'dup': [dp],
                    'device_group_union': [device_groups[stage_id]],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': [tp]},
                    'dup': [dp],
                    'device_group_union': [device_groups[stage_id]],
                    'type': 'variable'
                }
            }
        }
    return ds_parallel_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_size', type=str, default='7b', help='size of llama, 7b or 13b.'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=8, help='num of gpus.'
    )
    parser.add_argument(
        '--dp', type=int, default=2, help='dp.'
    )
    parser.add_argument(
        '--tp', type=int, default=2, help='tp.'
    )
    parser.add_argument(
        '--pp', type=int, default=2, help='pp.'
    )
    parser.add_argument(
        '--zero', action='store_true', help='use zero or not.'
    )
    # parser.add_argument(
    #     '--save_folder', type=str, default='./'
    # )
    args = parser.parse_args()
    if args.model_size == '3b':
        num_layers = 16
    elif args.model_size == '7b':
        num_layers = 32
    elif args.model_size == '13b':
        num_layers = 40
    elif args.model_size == '32b':
        num_layers = 60
    else:
        print('now only support 7b or 13b or 32b!')
        assert False, f'error! get model size {args.model_size}!'
        
    assert args.dp * args.tp * args.pp == args.num_gpus, \
            f'dp * tp * pp = {args.dp * args.tp * args.pp} is not equal to num_gpus {args.num_gpus}!'
    ds_parallel_config = generate_llama_3d_config(num_layers, args.num_gpus, args.dp, args.tp, args.pp, args.zero)
    save_folder = f'./ds_parallel_config/gpus{args.num_gpus}/{args.model_size}'
    file_name = f'dp{args.dp}_tp{args.tp}_pp{args.pp}.json'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)
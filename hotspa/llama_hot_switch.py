import os
import signal
import hetu as ht
from hetu_llama_multi_ds_parallel_symbolic_sp import LLamaLMHeadModel
from hetu.nn.modules.parallel_multi_ds import config2ds
from llama_config import LLaMAConfig
from data_utils import LLaMAJsonDataset, get_mask_and_position_ids,\
    build_dist_data_loader, build_normal_data_loader
import numpy as np
import time
import argparse
import json
import socket
import pynvml
from queue import Queue

local_device = None
all_devices = None
start_time = 0
end_time = 0

def distributed_init():
    hostname = socket.gethostname()
    # os.environ['HETU_LOCAL_HOSTNAME'] = os.environ['HOSTNAME']
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname

    global local_device, all_devices
    ht.init_comm_group(8)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')
    # used for debug
    # ptvsd.enable_attach(address =('127.0.0.1', 4000 + all_devices.get_index(local_device)))
    # ptvsd.wait_for_attach()

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    print(f'{local_device}: load ds_parallel_config from: {args.ds_parallel_config}')
    if ',' in args.ds_parallel_config:
        config_paths = args.ds_parallel_config.split(',')
    else:
        config_paths = args.ds_parallel_config.split()
    assert len(config_paths) == args.num_strategy, \
      f'ds_parallel_config num should equal to num_strategy {args.num_strategy}'
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = json.load(open(config_path, 'r'))
        # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp2_pp2.json', 'r'))
        # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp4.json', 'r'))
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
        ds_parallel_configs.append(ds_parallel_config)
    return ds_parallel_configs

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

def train_dataset_provider(args):
    """Build train dataset."""
    train_dataset = LLaMAJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.seq_length,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file)
    return train_dataset

def get_position_ids(gbs_per_dp, seq_len): 
    position_ids = np.arange(0, seq_len, dtype=np.int64) # [1, seq_len]
    position_ids = np.tile(position_ids, [gbs_per_dp, 1]) # [dp_size, seq_len]
    return position_ids

# get micro batch for cur dp rank
def dist_train_data_iterator(dataset, consumed_samples, mbs, dp_rank, dp_size):
    # print(f'new dataloader: consumed_samples = {consumed_samples}')
    dist_dataloader = build_dist_data_loader(dataset, consumed_samples, mbs, dp_rank, dp_size)
    dist_train_data_iter = iter(dist_dataloader)
    return dist_train_data_iter

# get global batch for all dp ranks
def normal_train_data_iterator(dataset, consumed_samples, global_batch_size):
    normal_dataloader = build_normal_data_loader(dataset, consumed_samples, global_batch_size)
    normal_train_data_iter = iter(normal_dataloader)
    return normal_train_data_iter

class Bucket:
    def __init__(self, max_seq_len, min_seq_len, pad_token):
        self._max_seq_len = max_seq_len + 1 # will shift 1 bit for <input, label>
        self._min_seq_len = min_seq_len + 1
        self._pad_token = pad_token
        self._batch = []
        self._packed_batch = []
        self._empty = False
    
    def add_data(self, padded_sequence, valid_tokens):
        if valid_tokens > self._min_seq_len and valid_tokens <= self._max_seq_len:
            self._batch.append(padded_sequence[:valid_tokens])
            # print(f'bucket max {self._max_seq_len}, min {self._min_seq_len} add: {valid_tokens}')
            return True
        else:
            return False

    def pack_data(self):
        is_visited = set()
        for i in range(len(self._batch)):
            if i in is_visited:
                continue
            cur_seq = [self._batch[i]]
            cur_len = len(self._batch[i])
            is_visited.add(i)
            for j in range(i + 1, len(self._batch)):
                if j not in is_visited and cur_len + len(self._batch[j]) <= self._max_seq_len:
                    cur_seq.append(self._batch[j])
                    cur_len += len(self._batch[j])
                    is_visited.add(j)
            # print(f'bucket max {self._max_seq_len}, min {self._min_seq_len} packed into: {cur_len}')
            if cur_len < self._max_seq_len:
                cur_seq.append(np.array([self._pad_token] * (self._max_seq_len - cur_len)))
            self._packed_batch.append(np.concatenate(cur_seq))
        if len(self._packed_batch) > 0:
            self._packed_batch = np.stack(self._packed_batch)
            
    def make_dummy_padded_batch(self):
        # actually do nothing when exists empty bucket
        self._max_seq_len = 129
        self._packed_batch = [np.array([self._pad_token] * self._max_seq_len)] * 8
        self._empty = True

    def packed_batch_size(self):
        return len(self._packed_batch)

    def real_batch_size(self):
        return len(self._batch)

    def packed_batch(self):
        return self._packed_batch

    def max_seq_len(self):
        return self._max_seq_len - 1
    
    def min_seq_len(self):
        return self._min_seq_len - 1

    def empty(self):
        return self._empty

def sort_and_pack_for_global_batch(global_batch, pad_token, bucket_sizes=[32768, 16384, 4096, 0]):
    non_pad_counts = np.sum(global_batch != pad_token, axis=1)
    # print(f'non_pad_counts = {non_pad_counts}, sum = {np.sum(non_pad_counts)}')
    sorted_indices = np.argsort(-non_pad_counts)
    sorted_global_batch = global_batch[sorted_indices]
    sorted_valid_tokens = non_pad_counts[sorted_indices]

    buckets = []
    assert len(bucket_sizes) >= 2, 'len(bucket size) must >= 2'
    left, right = 0, 1
    bucket = Bucket(bucket_sizes[left], bucket_sizes[right], pad_token)
    buckets.append(bucket)
    for padded_sequence, valid_tokens in zip(sorted_global_batch, sorted_valid_tokens):
        while (not bucket.add_data(padded_sequence, valid_tokens)) and (right < len(bucket_sizes) - 1):
            left += 1
            right += 1
            bucket = Bucket(bucket_sizes[left], bucket_sizes[right], pad_token)
            buckets.append(bucket)

    for bucket in buckets:
        bucket.pack_data()

    return buckets

def get_micro_batches_for_bucket_this_rank(dataset, consumed_samples, bucket_gbs, bucket_mbs, dp_rank, dp_size):
    assert bucket_gbs % (bucket_mbs * dp_size) == 0, \
        f'gbs {bucket_gbs} must be divided by (mbs {bucket_mbs} * dp {dp_size}) in this bucket!'
    bucket_dataloader = build_dist_data_loader(dataset, consumed_samples, bucket_mbs, dp_rank, dp_size)
    bucket_data_iter = iter(bucket_dataloader)
    bucket_num_micro_batches = bucket_gbs / (bucket_mbs * dp_size)

    bucket_micro_batches = []
    for _ in range(bucket_num_micro_batches):
        bucket_micro_batch = next(bucket_data_iter)
        bucket_micro_batches.append(bucket_micro_batch)
    # [num_micro_batches, micro_batch_size, max_seq_len + 1]
    bucket_micro_batches = np.concatenate(bucket_micro_batches, axis=0)
    return bucket_micro_batches


def profile_memory(device_index = 0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
    print("Device", device_index, ":", device_name)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = memory_info.total / 1024 / 1024
    used_memory = memory_info.used / 1024 / 1024
    free_memory = memory_info.free / 1024 / 1024
    print("Total Memory:", total_memory, "MiB")
    print("Used Memory:", used_memory, "MiB")
    print("Free Memory:", free_memory, "MiB")
    
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None

def pretrain(args):
    ds_parallel_configs = read_ds_parallel_config(args)
    num_strategy = len(ds_parallel_configs)
    args.bucket_sizes = [int(s) for s in args.bucket_sizes.split()]

    config = LLaMAConfig(vocab_size=args.vocab_size, 
                       n_positions=args.seq_length,
                       n_ctx=args.seq_length,
                       n_embd=args.hidden_size,
                       ffn_hidden_size=args.ffn_hidden_size,
                       n_layer=args.num_hidden_layers, 
                       n_head=args.num_attention_heads, 
                       seq_len=args.seq_length,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       use_flash_attn=args.use_flash_attn,
                       )
    
    # simple check for llama blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['llama']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == config.num_hidden_layers-1, \
        f"llama blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    model = LLamaLMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)
    
    input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    # mbs_times_dp = args.global_batch_size // config.num_micro_batches
    dp_size = input_ds_hierarchy[0].get(0).get_dim(0)
    mbs_times_dp = args.micro_batch_size * dp_size # dummy shape, will re-deduce when after sequence packing
        
    # todo: assign multi dg_hierarchy, and instantiate only one placement_group
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
    position_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
    token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')

    config.mbs_times_dp_symbol = ht.IntSymbol(mbs_times_dp)
    config.seq_len_symbol = input_ids.symbolic_shape[1]

    print(f'{local_device}: build model begin...')
    loss = model(input_ids=input_ids,
                 position_ids=position_ids,
                 # attention_mask=attention_mask,
                 token_type_ids=token_type_ids,
                 labels=masked_lm_labels)
    print(f'{local_device}: build model end...')

    loss_mean = loss

    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss_mean)
    print(f'{local_device}: optimizer minimize end...')
    
    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    data_iter = normal_train_data_iterator(train_dataset, 0, args.global_batch_size)
    print(f'{local_device}: build dataset end...')

    # each strategy has the seperate gbs and mbs for bucket.
    # the sum of gbs for all strategies in one step is equal to mini-batch size
    def run_plan(epoch = 0,
                 step = 0,
                 consumed_samples = 0,
                 bucket = None,
                 strategy_id = 0, 
                 run_level = 0):
        global start_time, end_time

        input_ds_union = input_ds_hierarchy[strategy_id]
        input_device_group_union = input_dg_hierarchy[strategy_id]
        label_ds_union = label_ds_hierarchy[strategy_id]
        label_device_group_union = label_dg_hierarchy[strategy_id]
        assert input_ds_union.hetero_dim == -3, "input hetero dim unsupported"
        assert label_ds_union.hetero_dim == -3, "label hetero dim unsupported"

        # device in same dp_group will read the same batch data, idx=-1 means no need to read data
        dup_group_idx, dup_group_num = -1, -1
        input_union_idx, input_device_group = get_dg_from_union(local_device, input_device_group_union)
        label_union_idx, label_device_group = get_dg_from_union(local_device, label_device_group_union)
        if input_device_group != None:
            local_device_idx = input_device_group.get_index(local_device)
            dup_group_idx = input_union_idx
            dup_group_num = len(input_device_group_union)
            if input_ds_union.hetero_dim == -3:
                dup_group_idx = input_ds_union.get(0).get_dup_group_index(local_device_idx)
                dup_group_num = input_ds_union.get(0).get_dim(0)
        elif label_device_group != None:
            local_device_idx = label_device_group.get_index(local_device)
            dup_group_idx = label_union_idx
            dup_group_num = len(label_device_group_union)
            if label_ds_union.hetero_dim == -3:
                dup_group_idx = label_ds_union.get(0).get_dup_group_index(local_device_idx)
                dup_group_num = label_ds_union.get(0).get_dim(0)
        else:
            dup_group_num = len(input_device_group_union)
            if input_ds_union.hetero_dim == -3:
                dup_group_num = input_ds_union.get(0).get_dim(0)
            # raise RuntimeError(f"device {local_device} not in input_device_group or label_device_group!")

        dp_rank = dup_group_idx
        dp_size = dup_group_num
        micro_batch_size = 1 # already packed
        global_batch_size = bucket.packed_batch_size() // dp_size * dp_size
        if global_batch_size == 0: # empty bucket, just run a dummy batch with full padded
            bucket.make_dummy_padded_batch()
            global_batch_size = dp_size
        seq_len = bucket.max_seq_len()
        gbs_per_dp = global_batch_size // dp_size
        mbs_times_dp = micro_batch_size * dp_size
        assert global_batch_size % mbs_times_dp == 0, \
            f'gbs {global_batch_size} must could be divided by mbs {micro_batch_size} * dp {dp_size}'
        num_micro_batches = global_batch_size // mbs_times_dp
                
        config.mbs_times_dp_symbol.set_data(mbs_times_dp)
        config.seq_len_symbol.set_data(seq_len)

        # profile_memory()

        # load data for each dp
        if dp_rank != -1:
            packed_batch = bucket.packed_batch()
            micro_batches = []
            for begin_idx in range(dp_rank * micro_batch_size, global_batch_size, mbs_times_dp):
                micro_batches.append(packed_batch[begin_idx: begin_idx+micro_batch_size])
            micro_batches = np.stack(micro_batches) # [num_micro_batches, micro_batch_size, max_seq_len + 1]
            micro_batches = micro_batches.reshape(gbs_per_dp, -1)[:, :seq_len+1] # [gbs_per_dp, seq_len + 1]
            labels = micro_batches[:, 1:] # [gbs_per_dp, seq_len]
            tokens = micro_batches[:, :-1] # [gbs_per_dp, seq_len]
            _attention_mask, _position_ids = get_mask_and_position_ids(tokens, train_dataset.encoder.pad_id())
            _token_type_ids = np.zeros([gbs_per_dp, seq_len])

            feed_dict = {
                input_ids: tokens.astype(np.int64),
                position_ids: _position_ids.astype(np.int64), 
                token_type_ids: _token_type_ids.astype(np.int64),
                # attention_mask: _attention_mask.astype(np.int64),
                masked_lm_labels: labels.astype(np.int64),
            }
        else: # fake data; feed_dict={} will cause segment fault?
            feed_dict = {
                input_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                position_ids: get_position_ids(gbs_per_dp, seq_len).astype(np.int64), 
                token_type_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                # attention_mask: np.zeros([gbs_per_dp, seq_len]).astype(np.float32),
                masked_lm_labels: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
            }
        if run_level == ht.run_level("alloc") or (len(ds_parallel_configs) == 1 and run_level == ht.run_level("update")):
            start_time = time.time()
        if args.run_memory_experiment and step == 0:
            os.environ['HETU_MEMORY_LOG_FILE'] = args.memory_file
        try:
            results = train_op.graph.run(loss_mean, 
                                            [loss_mean, train_op], 
                                            feed_dict = feed_dict, 
                                            num_micro_batches = num_micro_batches, 
                                            cur_strategy_id = strategy_id,
                                            run_level = run_level,
                                            grad_scale = 1.0)
        except RuntimeError as e:
            print(e)
            os.killpg(0, signal.SIGTERM)
        if args.run_memory_experiment and step == 0:
            if 'HETU_MEMORY_LOG_FILE' in os.environ:
                del os.environ['HETU_MEMORY_LOG_FILE'] 
            return consumed_samples
        
        if run_level != ht.run_level("alloc"):
            consumed_samples += bucket.real_batch_size()

        if run_level == ht.run_level("update"):
            # NOTE: update new step end time
            end_time = time.time()
            if label_device_group != None:
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")

        return consumed_samples
    
    # static parallelism strategy
    def test_static(): 
        assert num_strategy == 1, 'parallelism strategy num must == 1 when use static strategy!'
        strategy_id = 0
        for epoch in range(args.epochs):
            consumed_samples = 0 # should be reset when run next epoch
            for step in range(args.steps):
                global_batch = next(data_iter).numpy()
                buckets = sort_and_pack_for_global_batch(global_batch, train_dataset.pad_id(), args.bucket_sizes)
                assert len(buckets) == 1, 'static strategy only support one bucket!'
                consumed_samples = run_plan(epoch = epoch,
                                            step = step,
                                            consumed_samples = consumed_samples, 
                                            bucket = buckets[0],
                                            strategy_id = strategy_id, 
                                            run_level = ht.run_level("update"))
            
    # hot switching
    def test_switch(): 
        # assert num_strategy > 1, 'parallelism strategy num must > 1 when enable hot switching!'
        print(f'{local_device}: the first step may take a few minutes to [generate topo & execution plan], please be patient...')
        for epoch in range(args.epochs):
            consumed_samples = 0 # should be reset when run next epoch
            for step in range(args.steps):
                # group and packing
                global_batch = next(data_iter).numpy()
                buckets = sort_and_pack_for_global_batch(global_batch, train_dataset.pad_id(), args.bucket_sizes)
                # enable parallelism hot switching
                for strategy_id in range(num_strategy + 1):
                    alloc_run_level = ht.run_level("alloc")
                    grad_run_level = ht.run_level("grad")
                    update_run_level = ht.run_level("update")
                    if strategy_id == 0:
                        cur_run_level = alloc_run_level
                        bucket = buckets[0]
                    elif strategy_id == len(ds_parallel_configs): # init graph
                        strategy_id = 0
                        cur_run_level = update_run_level
                        bucket = buckets[0]
                    else:
                        cur_run_level = grad_run_level
                        bucket = buckets[num_strategy - strategy_id]
                    # run sperate strategy for bucket
                    consumed_samples = run_plan(epoch = epoch,
                                                step = step,
                                                consumed_samples = consumed_samples, 
                                                bucket = bucket, 
                                                strategy_id = strategy_id, 
                                                run_level = cur_run_level)
    
    if args.hot_switch:
        test_switch()
    else:
        test_static()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_memory_experiment", action="store_true", help="run memory experiment."
    )
    parser.add_argument(
        "--hot_switch", action="store_true", help='enable parallelism hot switching.'
    )
    parser.add_argument(
        "--test_func", action="store_true", help='test functional for parallelism hot switching.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--num_strategy", type=int, default=1, help="multi ds num"
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, 
        help="multi parallel strategy config file for hot switching"
    )
    parser.add_argument(
        "--bucket_sizes", default="32768 16384 4096 0", type=str, 
        help="multi bucket size for hot switching"
    )
    parser.add_argument(
        "--memory_file", default="", type=str, help="memory experiment result file"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="Training batch size global"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Training batch size each micro batch"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='llama vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='llama merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=-1, help="FFN hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=4, help="Number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of steps for each epoch",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    args = parser.parse_args()
    pynvml.nvmlInit()
    distributed_init()
    with ht.graph("define_and_run", num_strategy=args.num_strategy):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')
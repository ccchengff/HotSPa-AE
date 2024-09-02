import hetu
from .module import Module
import numbers

__all__ = [
    'ColumnParallelLinear', 
    'RowParallelLinear', 
    'ParallelEmbedding',
    'VocabParallelEmbedding',
    'ParallelLayerNorm',
]

def parallel_data_provider(global_data, ds, device_index):
    order, states = ds.order, ds.states
    local_map = hetu.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        local_data = local_data.take(range(start, stop), axis=dim)
    return local_data

def get_device_index(device_group):
    local_device = hetu.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index


class ParallelLayerNorm(Module):
    def __init__(self, normalized_shape, device_group, eps=1e-5, dtype=hetu.float32, name='ln'):
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.name = name
        self.device_group = device_group
        num_devices = device_group.num_devices
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1])
        device_index = get_device_index(device_group)
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, ds_dup, device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group=device_group, name=f'{name}_weight')
        self.bias = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                              self.normalized_shape, ds_dup, device_index, 
                                              dtype=dtype, requires_grad=True, device_group=device_group, name=f'{name}_bias')

    def forward(self, input_p):
        return hetu.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, self.eps, device_group=self.device_group, name=self.name)[0]

class ParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device_group, init_method='xavier_normal_', dtype=hetu.float32, name='embedding'):
        super(ParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device_group = device_group
        self.name = name
        num_devices = device_group.num_devices
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1])
        device_index = get_device_index(device_group)
        # embedding_table should not be splited in any dimension!
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], ds_dup, device_index, 
                                                       dtype=dtype, requires_grad=True, 
                                                       device_group=device_group, name=f'{name}_table')
    
    def forward(self, input_p):
        return hetu.embedding_lookup(self.embedding_table, input_p, device_group=self.device_group, name=self.name)
    
class VocabParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device_group, dp=1, fixed_vocab_start_index=True, init_method='xavier_normal_', dtype=hetu.float32, name='vocab_embedding'):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device_group = device_group
        self.dp = dp
        self.name = name
        num_devices = device_group.num_devices

        ds_split0_dup = hetu.DistributedStates(num_devices, {-1: num_devices//dp, 0: dp}, [0, -1]) # for data
        ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: dp, 0: num_devices//dp}, [-1, 0]) # for embedding table
        self.ds_map = {'split0_dup': ds_split0_dup, 'dup_split0': ds_dup_split0}
        device_index = get_device_index(device_group)
        dup_group_idx = ds_dup_split0.get_dup_group_index(device_index)
        if fixed_vocab_start_index:
            self.vocab_start_index = num_embeddings // (num_devices // dp) * dup_group_idx 

        # embedding_table was splited in vocab dimension
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], ds_dup_split0, device_index, 
                                                       dtype=dtype, requires_grad=True, 
                                                       device_group=device_group, name=f'{name}_table')
    
    def forward(self, input_p, vocab_start_index=None):
        if input_p.distributed_states.check_equal(self.ds_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.ds_map['split0_dup'])
            print('warning: vocab parallel embedding need extra communication for \
                  adapt input tensor distributed_states into split0_dup!')

        if vocab_start_index:
            input_offset = tensor_split0_dup - vocab_start_index
        else:
            input_offset = tensor_split0_dup - self.vocab_start_index
            
        lookup_split0_partial = hetu.embedding_lookup(self.embedding_table, input_offset, device_group=self.device_group, name=self.name)
        output = hetu.comm(lookup_split0_partial, self.ds_map['split0_dup'])
        return output

# todo: modify column and row parallel linear
# process: x->dup, w->split1 => y->split1 => y->dup
class ColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, device_group, dp=1,
                 bias=True, gather_output=True, init_method='xavier_normal_', dtype=hetu.float32, name='colp'):
        super(ColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_group = device_group
        self.dp = dp
        self.gather_output = gather_output
        self.name = name

        device_index = get_device_index(device_group)
        num_devices = device_group.num_devices
        
        ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: dp, 0: num_devices//dp}, [-1, 0]) # for weights
        ds_split0_dup = hetu.DistributedStates(num_devices, {-1: num_devices//dp, 0: dp}, [0, -1]) # for data
        self.ds_map = {'dup_split0': ds_dup_split0, 'split0_dup': ds_split0_dup}
        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [out_features, in_features], 
                                              ds_dup_split0, device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group=device_group, name=f'{name}_weight')
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                ds_dup_split0, device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_group=device_group, name=f'{name}_bias')
        else:
            self.bias = None
      
    def forward(self, input_p):
        if input_p.distributed_states.check_equal(self.ds_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.ds_map['split0_dup'])
            print(f'input_p.distributed_states={input_p.distributed_states}, split0_dup={self.ds_map["split0_dup"]}')
            print('warning: need extra communication for adapt input tensor distributed_states into split0_dup!')
        
        tensor_split01 = hetu.linear(tensor_split0_dup, self.weight, self.bias, trans_b=True, device_group=self.device_group, name=self.name)
        if not self.gather_output:
            output = tensor_split01
            # bias = self.bias # for fusion
        else:
            output = hetu.comm(tensor_split01, self.ds_map['split0_dup'])
            # bias = hetu.comm(self.bias, self.ds_map['split0_dup']) # for fusion
        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class RowParallelLinear(Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """
    def __init__(self, in_features, out_features, device_group, dp=1,
                 bias=True, init_method='xavier_normal_', dtype=hetu.float32, name='rowp'):
        super(RowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_group = device_group
        self.dp = dp
        self.name = name

        device_index = get_device_index(device_group)
        num_devices = device_group.num_devices

        self.partial = num_devices // dp
        ds_dup_split1 = hetu.DistributedStates(num_devices, {-1: dp, 1: num_devices//dp}, [-1, 1]) # for weight
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1]) # for bias
        ds_split01 = hetu.DistributedStates(num_devices, {0: dp, 1: num_devices//dp}, [0, 1]) # for data split in dimension 1
        ds_split0_dup = hetu.DistributedStates(num_devices, {-1: num_devices//dp, 0: dp}, [0, -1]) # for data reduce partial to dup
        self.ds_map = {'dup_split1': ds_dup_split1, 'dup': ds_dup, 'split01': ds_split01, 'split0_dup': ds_split0_dup}

        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [out_features, in_features], 
                                              ds_dup_split1, device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group=device_group, name=f'{name}_weight')        
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                ds_dup, device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_group=device_group, name=f'{name}_bias')
        else:
            self.bias = None

    def forward(self, input_p):
        if input_p.distributed_states.check_equal(self.ds_map['split01']):
            tensor_split01 = input_p
        else:
            tensor_split01 = hetu.comm(input_p, self.ds_map['split01'])

        tensor_split0_partial = hetu.linear(tensor_split01, self.weight, trans_b=True, device_group=self.device_group, name=self.name)
        tensor_split0_dup = hetu.comm(tensor_split0_partial, self.ds_map['split0_dup'])
        output = tensor_split0_dup + self.bias if self.bias is not None else tensor_split0_dup
        return output
import hetu
import os
import torch
import numpy as np


WEIGHTS_NAME = 'hetu_pytorch_model.bin'


def retrieve_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def change_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_heads * num_splits * hidden_size, :]
    # The original layout of param tensor is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    original_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
    param = param.view(*original_shape)
    param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def load_state_dict(checkpoint_file):
    
    return torch.load(checkpoint_file, map_location="cpu")


def load_checkpoint(model, path, config=None, local_device=None):
    
    # Load from a PyTorch checkpoint
    # TODO: more than one file to load (if the model is quite big)
    archive_file = os.path.join(
        path, WEIGHTS_NAME
    )
    
    # Tensor -> Numpy
    state_dict = load_state_dict(archive_file)
    for k in state_dict:
        # TODO: maybe implement it elsewhere
        # qkv_dense的weight和bias原先是[3 * num_heads * hidden_size, :]
        # 要先转化成为[num_heads * 3 * hidden_size, :]才可以
        # 因为会在num_heads这一维度上进行distributed tensor的split切分
        if "qkv_dense" in k:
            assert config != None, "There should be a config when using qkv_dense."
            state_dict[k] = change_query_key_value_ordering(state_dict[k], 
                                                            3, 
                                                            config.num_attention_heads, 
                                                            config.hidden_size // config.num_attention_heads)
        state_dict[k] = state_dict[k].numpy()
        
    # Time to load the checkpoint
    model.load_state_dict(state_dict, local_device=local_device)
    
    return model
    
    
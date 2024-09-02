import hetu
from hetu import Tensor
from .module import Module
import math

from typing import Any

__all__ = [
    'Embedding', 
]

class Embedding(Module):
    
    def __init__(self, num_embeddings, embedding_dim, device_group = None) -> None:
        with hetu.graph("define_and_run"):
            super(Embedding, self).__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = hetu.nn.functional.xavier_normal_([num_embeddings, embedding_dim], requires_grad=True, device_group = device_group)
    
    def forward(self, input: Tensor) -> Tensor:
        return hetu.embedding_lookup(self.weight, input)

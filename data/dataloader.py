import jax
import jax.numpy as jnp
import numpy as np
import pickle
import chex
from core.memory.replay_memory import BaseExperience

class DataLoader:
    def __init__(self, data_path: str, metadata_path: str, template_experience: BaseExperience, seed: int=42, discount: float=1.0):
        self.data_path = data_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        self.template_experience = template_experience
        self.epoch = 0
        self.next_idx = 0
        self.permutation = np.random.RandomState(seed).permutation(self.metadata['shape'][0])
        self.discount = discount
    def sample(self, batch_size: int) -> chex.ArrayTree:
        data = np.memmap(self.data_path, dtype=self.metadata['dtype'], mode='r', shape=self.metadata['shape'])
        indices = self.permutation[self.next_idx : self.next_idx + batch_size]
        raw_batch = data[indices]
        self.next_idx += batch_size
        if self.next_idx >= self.metadata['shape'][0]:
            self.epoch += 1
            self.next_idx = 0
            self.permutation = np.random.permutation(self.metadata['shape'][0])
        batch = jax.tree_util.tree_map(
            lambda x: jnp.zeros((batch_size, *x.shape), dtype=x.dtype),
            self.template_experience
        )
        return batch
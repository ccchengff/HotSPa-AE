"""Dataloaders."""

import torch

def build_dist_data_loader(dataset, consumed_samples, micro_batch_size, dp_rank, dp_size):
    """
        Buld dataloader given an input dataset.
        args: mbs, num_workers
    """

    if dataset is None:
        return None

    # batch sampler
    batch_sampler = LLaMADistBatchSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       shuffle=False,
                                       num_workers=0, # num_workers>0 exists bugs with mpirun
                                       pin_memory=False)

class LLaMADistBatchSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

# global batch -> [gbs_0, gbs_1, gbs_2, ..., gbs_n-1] for n strategy group
# gbs_i -> num_micro_batches_i x mbs_i x DP_i
# read one micro_batch in gbs_i, and return the sliced part for cur_dp_rank
# consumed_samples: gbs_0 + gbs_1 + ... + gbs_i-1

def build_normal_data_loader(dataset, consumed_samples, global_batch_size):
    """
        Buld dataloader given an input dataset.
    """

    if dataset is None:
        return None

    # batch sampler
    batch_sampler = LLaMANormalSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        global_batch_size=global_batch_size)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       shuffle=False,
                                       num_workers=0, # num_workers>0 exists bugs with mpirun
                                       pin_memory=False)

# directly return the whole global batch, will be split into chunks later
class LLaMANormalSampler:

    def __init__(self, total_samples, consumed_samples, 
                 global_batch_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.global_batch_size:
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch
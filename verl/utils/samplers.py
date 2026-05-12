from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import torch

class GroupedBatchSampler(BatchSampler):
   
    def __init__(self, base_len: int, n: int, batch_size: int, shuffle: bool = True, drop_last: bool = True, seed: int = 42):
        assert batch_size % n == 0, f"batch_size({batch_size}) must be multiple of n({n})"
        self.base_len = base_len
        self.n = n
        self.batch_size = batch_size
        self.groups_per_batch = batch_size // n
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._build_base_sampler()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._build_base_sampler()

    def _build_base_sampler(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.base_sampler = RandomSampler(range(self.base_len), generator=g)
        else:
            self.base_sampler = SequentialSampler(range(self.base_len))

    def __iter__(self):
        buf = []
        for base_idx in self.base_sampler:
            for r in range(self.n):
                buf.append(base_idx * self.n + r)
            if len(buf) == self.batch_size:
                yield buf
                buf = []
        if len(buf) > 0 and not self.drop_last:
            yield buf

    def __len__(self):
        total_groups = self.base_len
        batches = total_groups // self.groups_per_batch
        if not self.drop_last and (total_groups % self.groups_per_batch != 0):
            batches += 1
        return batches

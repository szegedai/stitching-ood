from typing import Iterable, Any, Optional

import torch
from torch.utils.data import Sampler

# Modified from https://discuss.pytorch.org/t/new-subset-every-epoch/85018

class MultiEpochDisjointSubsetSampler(Sampler):
    r"""Samples the entire dataset in random disjoint subsets across multiple epochs.
    """

    def __init__(self, data_source: Iterable[Any], n_samples: Optional[int] = None):
        """"""

        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples should be a positive integer value or None, but got n_samples={n_samples}")

        self.data_source = data_source
        self._n_samples = n_samples

        self.permute()

    def permute(self):
        """"""

        self.perm = torch.randperm(len(self.data_source), dtype=torch.int64)
        self.n_it = 0

    @property
    def n_samples(self):
        """"""

        if self._n_samples is None:
            return len(self.data_source)
        return self._n_samples

    def __iter__(self):
        """"""

        indices = self.perm[self.n_it * self.n_samples : (self.n_it + 1) * self.n_samples]
        self.n_it += 1

        if self.n_it * self.n_samples >= len(self.data_source):
            self.permute()

        return iter(indices)

    def __len__(self):
        return self.n_samples

import torch
import torch.utils.data as data

from .database import NASBenchDataBase
from .tensorize import tensorize_fn


class NASBench(data.Dataset):
    def __init__(self, database: NASBenchDataBase, seed):
        self.database = database
        g = torch.Generator()
        g.manual_seed(seed)
        self.proxy = torch.randperm(self.database.size, generator=g).tolist()

    def __getitem__(self, i):
        arch = self.database.items[self.proxy[i]]

        maxtrix, ops = tensorize_fn(arch)

        validation_accuracy = arch.validation_accuracy
        test_accuracy = arch.test_accuracy

        return maxtrix, ops, validation_accuracy, test_accuracy

    def __len__(self):
        return len(self.proxy)

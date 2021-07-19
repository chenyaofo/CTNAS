import torch
import torch.utils.data as data

# from core.genotypes import Genotype
from .database import CommonNASDataBase
from .tensorize import tensorize_fn
from core.utils import mean


class CommonNAS(data.Dataset):
    def __init__(self, database: CommonNASDataBase, seed):
        self.database = database
        g = torch.Generator()
        g.manual_seed(seed)
        self.proxy = torch.randperm(self.database.size, generator=g).tolist()

    def __getitem__(self, i):
        # print(i, self.proxy[i], len(self.database.items))
        arch = self.database.items[self.proxy[i]]

        normal_matrix, normal_ops, reduced_matrix, reduced_ops = tensorize_fn(arch)

        validation_accuracy = mean(arch.validation_accuracy[-10:])
        # validation_accuracy = max(arch.validation_accuracy)

        return normal_matrix, normal_ops, reduced_matrix, reduced_ops, validation_accuracy, validation_accuracy

    def __len__(self):
        return len(self.proxy)

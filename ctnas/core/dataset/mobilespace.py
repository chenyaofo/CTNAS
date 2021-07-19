import torch
import torch.utils.data as data

# from core.genotypes import Genotype
from .database import MBSpaceDataBase
from .seq2arch import seq2arch_fn
from .tensorize import tensorize_fn
from core.utils import mean


class MBSpaceDataset(data.Dataset):
    def __init__(self, database: MBSpaceDataBase, seed):
        self.database = database
        g = torch.Generator()
        g.manual_seed(seed)
        self.proxy = torch.randperm(self.database.size, generator=g).tolist()

    def __getitem__(self, i):
        arch = self.database.items[self.proxy[i]]

        tensor = tensorize_fn(seq2arch_fn(arch["arch"]))
        accuracy = arch["top1_acc"]

        return tensor, accuracy, accuracy

    def __len__(self):
        return len(self.proxy)

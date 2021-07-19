import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from .nasbench import DUMMY, INPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT
# from .nasbench import VALID_OPERATIONS
from .nasbench import MAX_NODES
from .nasbench import NASBenchDataBase, Architecture

op2embedding = {
    DUMMY: 0,
    INPUT: 1,
    CONV1X1: 2,
    CONV3X3: 3,
    MAXPOOL3X3: 4,
    OUTPUT: 5,
}

embedding2op = {v: k for k, v in op2embedding.items()}


def tensorize(arch: Architecture, batch=False, device="cpu"):
    matrix = torch.from_numpy(arch.matrix).float().transpose(0, 1)
    n_ops = len(arch.ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES-n_ops
        ops = arch.ops + [DUMMY] * diff
        matrix = F.pad(matrix, [0, diff, 0, diff])
    else:
        ops = arch.ops
    ops = torch.tensor([op2embedding[item] for item in ops], dtype=torch.long)
    if batch:
        matrix, ops = matrix.unsqueeze(dim=0), ops.unsqueeze(dim=0)
    matrix, ops = matrix.to(device=device), ops.to(device=device)
    return matrix, ops


def tensor2arch(matrix, ops):
    matrix = matrix.clone().cpu().transpose(0, 1).numpy().astype(np.int32)

    ops = ops.tolist()
    ops = [embedding2op[item] for item in ops]
    ops = [item for item in ops if item != DUMMY]

    extraneous = list(range(len(ops), MAX_NODES))

    matrix = np.delete(matrix, extraneous, axis=0)
    matrix = np.delete(matrix, extraneous, axis=1)
    # print(matrix)
    # print(ops)
    arch = Architecture.from_spec(matrix, ops)
    return arch


class NASBench(data.Dataset):
    def __init__(self, nasbench: NASBenchDataBase, seed):
        self.nasbench = nasbench
        g = torch.Generator()
        g.manual_seed(seed)
        self.proxy = torch.randperm(self.nasbench.size, generator=g).tolist()

    def __getitem__(self, i):
        arch = self.nasbench.archs_list[self.proxy[i]]

        maxtrix, ops = tensorize(arch)

        validation_accuracy = arch.validation_accuracy
        test_accuracy = arch.test_accuracy

        return (maxtrix, ops), validation_accuracy, test_accuracy

    def __len__(self):
        return len(self.proxy)


class CachedSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.cache = [self.dataset[i] for i in self.indices]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.indices)

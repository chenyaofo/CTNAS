import torch
import torch.nn.functional as F
from ..architecture.nasbench import *

op2id = {
    DUMMY: 0,
    INPUT: 1,
    CONV1X1: 2,
    CONV3X3: 3,
    MAXPOOL3X3: 4,
    OUTPUT: 5,
}

id2op = {v: k for k, v in op2id.items()}


def nasbench_tensorize(arch: NASBenchArchitecture, batch=False, device="cpu"):
    matrix = torch.from_numpy(arch.matrix).float().transpose(0, 1)
    n_ops = len(arch.ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES-n_ops
        ops = arch.ops + [DUMMY] * diff
        matrix = F.pad(matrix, [0, diff, 0, diff])
    else:
        ops = arch.ops
    ops = torch.tensor([op2id[item] for item in ops], dtype=torch.long)
    if batch:
        matrix, ops = matrix.unsqueeze(dim=0), ops.unsqueeze(dim=0)
    matrix, ops = matrix.to(device=device), ops.to(device=device)
    return matrix, ops


def nasbench_tensor2arch(matrix, ops):
    matrix = matrix.clone().cpu().transpose(0, 1).numpy().astype(np.int32)

    ops = ops.tolist()
    ops = [id2op[item] for item in ops]
    ops = [item for item in ops if item != DUMMY]

    extraneous = list(range(len(ops), MAX_NODES))

    matrix = np.delete(matrix, extraneous, axis=0)
    matrix = np.delete(matrix, extraneous, axis=1)

    arch = NASBenchArchitecture.from_spec(matrix, ops)
    return arch

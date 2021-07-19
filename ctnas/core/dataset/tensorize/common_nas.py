
import torch

from ..architecture.common_nas import *

op2embedding = {
    INPUT0: 0,
    INPUT1: 1,
    NONE: 2,
    SKIPCONNECT: 3,
    SEPCONV3x3: 4,
    SEPCONV5x5: 5,
    MAXPOOL3x3: 6,
    AVGPOOL3x3: 7,
    DILCONV3x3: 8,
    DILCONV5x5: 9,
}


def common_tensorize(arch: CommonNASArchitecture, batch=False, device="cpu"):
    normal_matrix = torch.from_numpy(arch.normal_matrix).float().transpose(0, 1)
    reduced_matrix = torch.from_numpy(arch.reduced_matrix).float().transpose(0, 1)

    normal_ops = torch.tensor([op2embedding[item] for item in arch.normal_ops], dtype=torch.long)
    reduced_ops = torch.tensor([op2embedding[item] for item in arch.reduced_ops], dtype=torch.long)

    if batch:
        normal_matrix, normal_ops = normal_matrix.unsqueeze(dim=0), normal_ops.unsqueeze(dim=0)
        reduced_matrix, reduced_ops = reduced_matrix.unsqueeze(dim=0), reduced_ops.unsqueeze(dim=0)

    normal_matrix, normal_ops = normal_matrix.to(device=device), normal_ops.to(device=device)
    reduced_matrix, reduced_ops = reduced_matrix.to(device=device), reduced_ops.to(device=device)
    return normal_matrix, normal_ops, reduced_matrix, reduced_ops

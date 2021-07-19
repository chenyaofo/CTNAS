import torch
import torch.nn as nn
from core.dataset.architecture.mobilespace import MBArchitecture, N_UNITS, DEPTHS


def _zero(layers, items, n_units=N_UNITS, n_layers=max(DEPTHS)):
    cnt = 0
    for n_unit in range(n_units):
        for n_layer in range(n_layers):
            items[cnt] = 0 if n_layer >= layers[n_unit] else items[cnt]
            cnt += 1
    return items


def mb_arch2tensor(arch: MBArchitecture, batch=False, device="cpu"):
    ks = _zero(arch.depths, arch.ks)
    ratios = _zero(arch.depths, arch.ratios)
    tensor = []
    for k in ks:
        if k == 0:
            tensor += [0, 0, 0]
        elif k == 3:
            tensor += [1, 0, 0]
        elif k == 5:
            tensor += [0, 1, 0]
        elif k == 7:
            tensor += [0, 0, 1]
        else:
            raise ValueError()
    for r in ratios:
        if r == 0:
            tensor += [0, 0, 0]
        elif r == 3:
            tensor += [1, 0, 0]
        elif r == 4:
            tensor += [0, 1, 0]
        elif r == 6:
            tensor += [0, 0, 1]
        else:
            raise ValueError()
    out = torch.tensor(tensor, dtype=torch.float)
    if batch:
        out = out.unsqueeze(dim=0)
    out = out.to(device=device)
    return out

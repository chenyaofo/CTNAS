from ..architecture import NASBenchArchitecture, CommonNASArchitecture, MBArchitecture
from .nasbench import nasbench_tensorize, nasbench_tensor2arch
from .common_nas import common_tensorize
from .mobilespace import mb_arch2tensor


def tensorize_fn(arch, device="cpu"):
    if isinstance(arch, NASBenchArchitecture):
        return nasbench_tensorize(arch, device=device)
    elif isinstance(arch, CommonNASArchitecture):
        return common_tensorize(arch, device=device)
    elif isinstance(arch, MBArchitecture):
        return mb_arch2tensor(arch, device=device)
    else:
        raise ValueError()

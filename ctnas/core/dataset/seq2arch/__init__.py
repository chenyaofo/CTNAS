from .nasbench import nasbench_seq2arch
from .common_nas import common_seq2arch, genotype2arch
from .mobilespace import str2arch


def seq2arch_fn(seq):
    if isinstance(seq, str):
        return str2arch(seq)
    else:
        if len(seq) == 1:
            if isinstance(seq[0], str):
                return str2arch(seq[0])
            else:
                return nasbench_seq2arch(seq[0])
        elif len(seq) == 2:
            return common_seq2arch(seq)
    raise ValueError()

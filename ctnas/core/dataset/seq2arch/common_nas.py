
import numpy as np
from core.genotypes import Genotype
from ..architecture.common_nas import *


def genotype2arch(items, n_nodes=4):
    matrix = np.zeros(shape=(2*n_nodes+2, 2*n_nodes+2), dtype=np.int32)

    virual_nodes = [list() for _ in range(n_nodes+2)]
    ops = [INPUT0, INPUT1]
    for i, (op_name, from_, to_) in enumerate(items):
        virual_nodes[to_].append(i+2)
        ops.append(op_name)

    for i, (op_name, from_, to_) in enumerate(items):
        if from_ < 2:
            matrix[from_][i+2] = 1
        else:
            for virual_from in virual_nodes[from_]:
                matrix[virual_from][i+2] = 1

    return matrix, ops


def common_seq2arch(seq):
    normal_seq, reduced_seq = seq
    geno = Genotype.from_ordinal_arch(ordinal_normal_arch=normal_seq, ordinal_reduced_arch=reduced_seq, primitives=PRIMITIVES)
    arch = CommonNASArchitecture()
    matrix, ops = genotype2arch(geno.named_normal_arch)
    arch.normal_matrix = matrix
    arch.normal_ops = ops
    matrix, ops = genotype2arch(geno.named_reduced_arch)
    arch.reduced_matrix = matrix
    arch.reduced_ops = ops
    return arch

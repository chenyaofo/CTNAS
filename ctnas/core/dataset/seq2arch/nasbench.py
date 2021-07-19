import numpy as np

from ..architecture.nasbench import *


def nasbench_seq2arch(seq):
    n_nodes = seq[0]
    matrix = np.zeros(shape=(n_nodes+2, n_nodes+2), dtype=np.int32)
    ops = [INPUT]
    rev = seq[1:2*n_nodes+1]
    for to_node, (from_node, op_type) in enumerate(zip(rev[::2], rev[1::2]), start=1):
        matrix[from_node][to_node] = 1
        ops.append(VALID_OPERATIONS[op_type])
    matrix[seq[2*n_nodes+1]][n_nodes+2-1] = 1
    rev = seq[2*n_nodes+2:]
    for (node0, node1) in zip(rev[::2], rev[1::2]):
        if node0 == node1:
            continue
        from_node = min(node0, node1)
        to_node = max(node0, node1)
        matrix[from_node][to_node] = 1
    ops.append(OUTPUT)
    arch = NASBenchArchitecture.from_spec(matrix, ops)
    return arch

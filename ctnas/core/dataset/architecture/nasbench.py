'''
Modified from https://github.com/google-research/nasbench

Apache License 2.0 text can be found at 
https://raw.githubusercontent.com/google-research/nasbench/master/LICENSE
'''
import dataclasses
import typing
import hashlib
import numpy as np


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
DUMMY = "dummy"
VALID_OPERATIONS = [CONV3X3, CONV1X1, MAXPOOL3X3]

MAX_VERTICES = 7
MAX_EDGES = 9
MAX_NODES = MAX_VERTICES
MAX_INTERMEDIATE_NODES = MAX_NODES - 2


def hash_fn(matrix, labeling):
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint


@dataclasses.dataclass
class NASBenchArchitecture:
    matrix: np.ndarray = None
    ops: typing.List[str] = None

    n_params: int = None

    training_time: float = None

    train_accuracy: float = None
    validation_accuracy: float = None
    test_accuracy: float = None

    hash_: str = None

    rank: int = None

    @property
    def n_ops(self):
        return len(self.ops)

    def hash(self):
        labeling = [-1] + [VALID_OPERATIONS.index(op) for op in self.ops[1:-1]] + [-2]
        return hash_fn(self.matrix, labeling)

    @classmethod
    def from_dict(cls, d):
        arch = NASBenchArchitecture()
        arch.__dict__.update(d)
        arch.matrix = np.array(arch.matrix)
        return arch

    def to_dict(self):
        return dataclasses.asdict(self)

    def _check_spec(self):
        """Checks that the model spec is within the dataset."""
        num_vertices = len(self.ops)
        num_edges = np.sum(self.matrix)

        if num_vertices > 7:
            raise ValueError()

        if num_edges > 9:
            raise ValueError()

        if self.ops[0] != 'input':
            raise ValueError()
        if self.ops[-1] != 'output':
            raise ValueError()
        for op in self.ops[1:-1]:
            if op not in VALID_OPERATIONS:
                raise ValueError()

    def _prune(self):
        num_vertices = np.shape(self.matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            self.ops.pop(index)

    @classmethod
    def from_spec(cls, matrix, ops):
        arch = NASBenchArchitecture()
        arch.matrix = matrix
        arch.ops = ops

        arch._prune()
        arch._check_spec()
        return arch

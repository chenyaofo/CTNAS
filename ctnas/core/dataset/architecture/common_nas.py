import dataclasses
import typing

import numpy as np

INPUT0 = "input0"
INPUT1 = "input1"
SEPCONV3x3 = "sep_conv_3x3"
SEPCONV5x5 = "sep_conv_5x5"
MAXPOOL3x3 = "max_pool_3x3"
AVGPOOL3x3 = "avg_pool_3x3"
NONE = "none"
SKIPCONNECT = "skip_connect"
DILCONV3x3 = "dil_conv_3x3"
DILCONV5x5 = "dil_conv_5x5"

PRIMITIVES = [
    SKIPCONNECT,
    SEPCONV3x3,
    SEPCONV5x5,
    NONE,
    DILCONV3x3,
    DILCONV5x5,
    MAXPOOL3x3,
    AVGPOOL3x3,
]


@dataclasses.dataclass
class CommonNASArchitecture:
    normal_matrix: np.ndarray = None
    normal_ops: typing.List[str] = None

    reduced_matrix: np.ndarray = None
    reduced_ops: typing.List[str] = None

    train_accuracy: float = None
    validation_accuracy: float = None

    @classmethod
    def from_dict(cls, d):
        arch = cls()
        arch.__dict__.update(d)
        arch.normal_matrix = np.array(arch.normal_matrix)
        arch.reduced_matrix = np.array(arch.reduced_matrix)
        return arch

    def to_dict(self):
        self.normal_matrix = self.normal_matrix.tolist()
        self.reduced_matrix = self.reduced_matrix.tolist()
        return dataclasses.asdict(self)

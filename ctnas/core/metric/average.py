import torch


class AverageMetric(object):
    def __init__(self):
        self.reset()

    def reset(self,) -> None:
        self._n = 0
        self._value = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self._value += value.item()
        elif isinstance(value, (int, float)):
            self._value += value
        else:
            raise ValueError("'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self._n += 1

    def compute(self) -> float:
        return self._value / (self._n+1e-15)

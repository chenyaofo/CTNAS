import torch


class MovingAverageMetric(object):
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.reset()

    def reset(self,) -> None:
        self._value = 0

    def update(self, value) -> None:
        if torch.is_tensor(value):
            v = value.item()
        elif isinstance(value, (int, float)):
            v = value
        else:
            raise ValueError("'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        if self._value is None:
            self._value = v
        else:
            self._value = self.alpha*self._value+(1-self.alpha)*v

    def compute(self) -> float:
        return self._value

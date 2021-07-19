import torch


class AccuracyMetric(object):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.n_correct = 0
        self.n_total = 0

    def update(self, targets, outputs):
        batch_size = targets.size(0)

        pred = (outputs > 0.5).long()
        correct = pred.eq(targets.long()).sum()

        self.n_correct += correct.item()
        self.n_total += batch_size

    def compute(self):
        return self.n_correct / (self.n_total+1e-15)

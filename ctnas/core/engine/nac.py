import random
import functools

import torch
import torch.nn.functional as F

from core.metric import AccuracyMetric, AverageMetric
from core.utils import *


def train(epoch, labeled_loader, pseudo_set, pseudo_ratio, nac, criterion, optimizer, report_freq=10):
    nac.train()
    accuracy = AccuracyMetric()
    loss_avg = AverageMetric()

    for iter_, (*arch, acc, _) in enumerate(labeled_loader, start=1):
        *arch0, acc0 = to_device(*arch, acc)
        *arch1, acc1 = shuffle(*arch0, acc0)
        targets = (acc0 > acc1).float()
        if pseudo_set is not None and pseudo_ratio != 0:
            batch_size = int(acc0.shape[0] * pseudo_ratio)
            pseudo_set_size = len(pseudo_set[0][0])
            index = random.sample(list(range(pseudo_set_size)), batch_size)
            un_arch0, un_arch1, pseudo_labels = pseudo_set
            un_arch0 = list_select(un_arch0, index)
            un_arch1 = list_select(un_arch1, index)
            pseudo_labels = pseudo_labels[index]
            # import ipdb; ipdb.set_trace()
            arch0 = concat(arch0, un_arch0)
            arch1 = concat(arch1, un_arch1)
            targets = torch.cat([targets, pseudo_labels], dim=0)

        optimizer.zero_grad()

        outputs = nac(arch0, arch1)

        loss = criterion(outputs, targets)

        loss_avg.update(loss)
        accuracy.update(targets, outputs)

        loss.backward()
        optimizer.step()

    logger.info(
        ", ".join([
            "TRAIN Complete",
            f"epoch={epoch:03d}",
            f"accuracy={accuracy.compute()*100:.4f}%",
            f"loss={loss_avg.compute():.4f}",
        ])
    )
    return accuracy.compute(), loss_avg.compute()


def evaluate(epoch, loader, nac):
    nac.eval()

    KTau = AverageMetric()

    for iter_, (*arch, acc, _) in enumerate(loader, start=1):
        *arch, acc = to_device(*arch, acc)

        KTau_ = compute_kendall_tau_AR(nac, arch, acc)

        KTau.update(KTau_)

        logger.info(
            ", ".join([
                "EVAL Complete" if iter_ == len(loader) else "EVAL",
                f"epoch={epoch:03d}",
                f"iter={iter_:03d}",
                f"KTau={KTau_:.4f}({KTau.compute():.4f})",
            ])
        )
    return KTau.compute()

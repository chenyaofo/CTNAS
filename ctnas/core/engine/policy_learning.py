
import os
import copy
import random
import functools

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard

from core.model import NAC
from core.metric import AverageMetric, MovingAverageMetric
from core.genotypes import Genotype
from core.dataset.database import DataBase
from core.dataset.architecture.common_nas import PRIMITIVES
from core.dataset.utils import ControllerDataset
from core.dataset.seq2arch import seq2arch_fn
from core.dataset.tensorize import tensorize_fn, nasbench_tensor2arch
from core.controller import NASBenchController
from core.config import args
from core.utils import *


def single_batchify(*items):
    return [item.unsqueeze(0) for item in items]


best_iters = -1


def train_controller(max_iter: int, database: DataBase,
                     entropy_coeff: float, grad_clip: int,
                     controller: NASBenchController, nac: NAC,
                     optimizer: optim.Optimizer, writer: tensorboard.SummaryWriter,
                     alternate_train, alternate_evaluate, random_baseline=False,
                     log_frequence: int = 10, search_space=None):
    controller.train()
    nac.eval()
    optimizer.zero_grad()

    policy_loss_avg = MovingAverageMetric()
    entropy_mavg = MovingAverageMetric()
    logp_mavg = MovingAverageMetric()
    score_avg = MovingAverageMetric()

    pseudo_architecture_set = None

    with torch.no_grad():
        *arch_seq, _, _ = controller(force_uniform=True)
        raw_arch = seq2arch_fn(arch_seq)
        baseline_arch = [tensorize_fn(raw_arch, device=device)]

    best_collect_archs = [arch_seq]

    for iter_ in range(max_iter):

        if iter_ % args.n_iteration_update_pseudoset == 0 and args.pseudo_ratio != 0:
            if pseudo_architecture_set is None:
                pseudo_architecture_set = \
                    generate_architecture_with_pseudo_labels(
                        nac, controller,
                        2*int(args.pseudo_ratio*args.train_batch_size),
                        int(args.pseudo_ratio*args.train_batch_size))
            else:
                pseudo_architecture_set = list_concat(
                    pseudo_architecture_set,
                    generate_architecture_with_pseudo_labels(
                        nac, controller,
                        2*args.n_sample_architectures, args.n_sample_architectures)
                )

            epoch = args.nac_epochs + iter_
            accuracy, rank_loss = alternate_train(epoch=epoch, pseudo_set=pseudo_architecture_set)
            writer.add_scalar("nac/train_accuracy", accuracy, epoch)
            writer.add_scalar("nac/loss", rank_loss, epoch)
            KTau = alternate_evaluate(epoch=epoch)
            writer.add_scalar("nac/ktau", KTau, epoch)

        *arch_seq, logp, entropy = controller()
        with torch.no_grad():
            sample_arch = [tensorize_fn(seq2arch_fn(arch_seq), device=device)]
            score = nac(batchify(sample_arch), batchify(baseline_arch))
            score = score.mean().item()

        policy_loss = -logp * score - entropy_coeff * entropy

        optimizer.zero_grad()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(controller.parameters(), grad_clip)
        policy_loss.backward()
        optimizer.step()

        policy_loss_avg.update(policy_loss)
        entropy_mavg.update(entropy)
        logp_mavg.update(logp)
        score_avg.update(score)

        if iter_ % log_frequence == 0:
            logger.info(
                ", ".join([
                    "Policy Learning",
                    f"iter={iter_:03d}",
                    f"policy loss={policy_loss_avg.compute():.4f}",
                    f"entropy={entropy_mavg.compute():.4f}",
                    f"logp={logp_mavg.compute():.4f}",
                ])
            )
            writer.add_scalar("policy_learning/loss", policy_loss_avg.compute(), iter_)
            writer.add_scalar("policy_learning/entropy", entropy_mavg.compute(), iter_)
            writer.add_scalar("policy_learning/logp", logp_mavg.compute(), iter_)
            writer.add_scalar("policy_learning/reward", score_avg.compute(), iter_)

        if iter_ % args.evaluate_controller_freq == 0:
            baseline_arch, best_collect_archs = derive(iter_, controller, nac, 10,
                                                       database, writer, best_collect_archs,
                                                       random_baseline, search_space)
            torch.save(controller.state_dict(), os.path.join(args.output,f"controller-{iter_}.path"))


def generate_architecture_with_pseudo_labels(nac, controller, total, k):
    with torch.no_grad():
        arch_seqs = [controller()[:-2] for _ in range(total)]
        sample_archs = [seq2arch_fn(seq) for seq in arch_seqs]
        arch0 = [tensorize_fn(arch, device=device) for arch in sample_archs]

        arch0 = batchify(arch0)
        if not isinstance(arch0, (list, tuple)):
            arch0 = [arch0]
        arch1 = shuffle(*arch0)
        # import ipdb; ipdb.set_trace()
        p = nac(arch0, arch1)
        select_p, index = torch.topk(p, k=k)

        arch0 = list_select(arch0, index)
        arch1 = list_select(arch1, index)
        labels = (select_p > 0.5).float()

    return arch0, arch1, labels


def derive(iter_, controller: NASBenchController, nac: NAC, n_derive: int,
           database, writer, best_collect_archs, random_baseline=False, search_space=None):
    controller.eval()
    with torch.no_grad():
        arch_seqs = [controller()[:-2] for _ in range(n_derive)]
        sample_archs = [seq2arch_fn(seq) for seq in arch_seqs]
        arch_tensor = [tensorize_fn(arch, device=device) for arch in sample_archs]

    if random_baseline:
        location = random.choice(list(range(len(arch_tensor))))
    else:
        outputs = cartesian_traverse(arch_tensor, arch_tensor, nac)
        outputs.fill_diagonal_(0)
        max_p, location = outputs.sum(dim=1).max(dim=0, keepdim=True)
        max_p = max_p.view([]).item() / n_derive
        location = location.view([]).item()

    if database is not None:
        arch = database.fetch_by_spec(sample_archs[location])
        writer.add_scalar("policy_learning/besttop", arch.rank/database.size*100, iter_)

    history_arch_seqs = arch_seqs + best_collect_archs
    history_arch_tensor = arch_tensor + [tensorize_fn(seq2arch_fn(arch), device=device) for arch in best_collect_archs]
    his_outputs = cartesian_traverse(history_arch_tensor, history_arch_tensor, nac)
    his_outputs.fill_diagonal_(0)
    his_max_p, his_location = his_outputs.sum(dim=1).max(dim=0, keepdim=True)
    his_max_p = his_max_p.view([]).item() / (n_derive-1)
    his_location = his_location.view([]).item()

    global best_iters
    if his_location < n_derive:
        best_iters = iter_

    if database is not None:
        his_best_arch = database.fetch_by_spec(seq2arch_fn(history_arch_seqs[his_location]))
        writer.add_scalar("policy_learning/history_top", his_best_arch.rank/database.size*100, iter_)

    if search_space == "nasbench":
        logger.info(
            ", ".join([
                "DERIVE",
                f"iters={iter_}",
                f"derive {n_derive} archs, the best arch id = {location:02d}, p={max_p*100:.2f}%",
                f"test acc={arch.test_accuracy*100:.2f}%",
                f"rank={arch.rank}/{database.size}({arch.rank/database.size*100:.4f}%)",
                f"history best sampled in {best_iters} iters",
                f"test acc={his_best_arch.test_accuracy*100:.2f}%",
                f"rank={his_best_arch.rank}/{database.size}({his_best_arch.rank/database.size*100:.4f}%)",
            ])
        )
    elif search_space == "darts":
        best_geno = Genotype.from_ordinal_arch(ordinal_normal_arch=arch_seqs[location][0],
                                               ordinal_reduced_arch=arch_seqs[location][1],
                                               primitives=PRIMITIVES)
        his_best_geno = Genotype.from_ordinal_arch(ordinal_normal_arch=history_arch_seqs[his_location][0],
                                                   ordinal_reduced_arch=history_arch_seqs[his_location][1],
                                                   primitives=PRIMITIVES)
        logger.info(
            ", ".join([
                f"DERIVE",
                f"iters={iter_}",
                f"derive {n_derive} archs, the best arch id = {location:02d}, p={max_p*100:.2f}%",
                f"genotype={best_geno}",
                f"history best sampled in {best_iters} iters",
                f"genotype={his_best_geno}",
            ])
        )
    elif search_space == "mobilespace":
        best_arch = arch_seqs[location]
        his_best_arch = history_arch_seqs[his_location]
        logger.info(
            ", ".join([
                f"DERIVE",
                f"iters={iter_}",
                f"derive {n_derive} archs, the best arch id = {location:02d}, p={max_p*100:.2f}%",
                f"best_arch={best_arch}",
                f"history best sampled in {best_iters} iters",
                f"his_best_arch={his_best_arch}",
            ])
        )
    os.makedirs(os.path.join(args.output, "controllers"), exist_ok=True)
    torch.save(controller.state_dict(), os.path.join(args.output, "controllers", f"controller-{iter_}.pth"))

    return [arch_tensor[location]], [history_arch_seqs[his_location]]

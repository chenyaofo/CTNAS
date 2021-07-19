import os
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard

from core.dataset import NASBenchDataBase, CommonNASDataBase
from core.dataset import NASBench, CommonNAS, CachedSubset

from core.model import NAC
from core.controller import NASBenchController, LargeSpaceController

from core.engine.nac import train, evaluate
from core.engine.policy_learning import train_controller
from core.config import args
from core.utils import logger, set_reproducible
from core.utils import device

if __name__ == "__main__":
    logger.info(args)
    set_reproducible(args.seed)
    writer = tensorboard.SummaryWriter(args.output)

    logger.info(f"Loading database from {args.data}...")
    nac = NAC(n_nodes=args.n_nodes if args.space == "nasbench" else args.n_nodes*2, n_ops=args.n_ops,
              ratio=2 if args.space == "nasbench" else 4,
              n_layers=args.n_layers, embedding_dim=args.embedding_dim).to(device=device)

    max_epochs = args.nac_epochs

    if args.space == "nasbench":
        database = NASBenchDataBase.from_file(args.data)
        dataset = NASBench(database=database, seed=args.seed)
        trainset = CachedSubset(dataset, list(range(args.trainset_size)))
        valset = CachedSubset(dataset, list(range(args.trainset_size, args.trainset_size+args.valset_size, 1)))
    else:
        database = CommonNASDataBase.from_file(args.data)
        dataset = CommonNAS(database=database, seed=args.seed)
        trainset = CachedSubset(dataset, list(range(args.trainset_size)))
        valset = CachedSubset(dataset, list(range(args.trainset_size, args.trainset_size+args.valset_size, 1)))

    train_loader = data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    criterion = nn.BCELoss()

    nac_optimizer = optim.Adam(nac.parameters(), lr=args.nac_lr, betas=(0.5, 0.999), weight_decay=5e-4)

    best_KTau = -1.0
    for epoch in range(1, max_epochs+1):
        accuracy, loss = train(epoch=epoch, labeled_loader=train_loader, pseudo_set=None,
                               pseudo_ratio=args.pseudo_ratio,
                               nac=nac, criterion=criterion, optimizer=nac_optimizer)

        writer.add_scalar("nac/train_accuracy", accuracy, epoch)
        writer.add_scalar("nac/loss", loss, epoch)

        if epoch % args.evaluate_nac_freq == 0:
            KTau = evaluate(epoch=epoch, loader=val_loader, nac=nac)
            state_dict = nac.state_dict()
            if KTau > best_KTau:
                best_KTau = KTau

            writer.add_scalar("nac/ktau", KTau, epoch)
            writer.add_scalar("nac/best_ktau", best_KTau, epoch)

            save_dir = os.path.join(args.output, "NAC")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(save_dir, f"nac-{epoch}.pth"))
    logger.info(f"Train NAC Complete, Best KTau={best_KTau:.4f}")

    # train controller
    Controller = NASBenchController if args.space == "nasbench" else LargeSpaceController
    controller = Controller(n_ops=args.n_ops, n_nodes=args.n_nodes,
                            hidden_size=args.hidden_size, temperature=args.temperature,
                            tanh_constant=args.controller_tanh_constant,
                            op_tanh_reduce=args.controller_op_tanh_reduce, device=device).to(device=device)
    controller_optimizer = optim.Adam(controller.parameters(), args.controller_lr,
                                      betas=(0.5, 0.999), weight_decay=5e-4)
    alternate_train = functools.partial(train, labeled_loader=train_loader,
                                        pseudo_ratio=args.pseudo_ratio,
                                        nac=nac, criterion=criterion, optimizer=nac_optimizer)
    alternate_evaluate = functools.partial(evaluate, loader=val_loader, nac=nac)
    train_controller(max_iter=args.pl_iters, entropy_coeff=args.entropy_coeff, grad_clip=args.controller_grad_clip,
                     controller=controller, nac=nac, optimizer=controller_optimizer, writer=writer, database=database if args.space == "nasbench" else None,
                     alternate_train=alternate_train, alternate_evaluate=alternate_evaluate, search_space=args.space)

# Train Ciresan's 6-layer deep MNIST network
# (from http://yann.lecun.com/exdb/mnist/)

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Callable, List

import autograd_lib
import globals as gl
# import torch
import scipy
import torch
import torch.nn as nn
import torchcontrib
import wandb
from attrdict import AttrDefault, AttrDict
from torch.utils.tensorboard import SummaryWriter

import util as u

import os
import argparse
from importlib import import_module
import shutil
import json

import torch
import torch.nn.functional as F
import wandb

# for line profiling
try:
    # noinspection PyUnboundLocalVariable
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


@profile
def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--wandb', type=int, default=0, help='log to weights and biases')
    parser.add_argument('--autograd_check', type=int, default=0, help='autograd correctness checks')
    parser.add_argument('--logdir', type=str, default='/tmp/runs/curv_train_tiny/run')

    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--bias', type=int, default=1, help="whether to add bias between layers")

    parser.add_argument('--layer', type=int, default=-1, help="restrict updates to this layer")
    parser.add_argument('--data_width', type=int, default=28)
    parser.add_argument('--targets_width', type=int, default=28)
    parser.add_argument('--hess_samples', type=int, default=1, help='number of samples when sub-sampling outputs, 0 for exact hessian')
    parser.add_argument('--hess_kfac', type=int, default=0, help='whether to use KFAC approximation for hessian')
    parser.add_argument('--compute_rho', type=int, default=0, help='use expensive method to compute rho')
    parser.add_argument('--skip_stats', type=int, default=0, help='skip all stats collection')

    parser.add_argument('--dataset_size', type=int, default=60000)
    parser.add_argument('--train_steps', type=int, default=100, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")

    parser.add_argument('--full_batch', type=int, default=0, help='do stats on the whole dataset')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--swa', type=int, default=0)
    parser.add_argument('--lmb', type=float, default=1e-3)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--stats_batch_size', type=int, default=10000)
    parser.add_argument('--stats_num_batches', type=int, default=1)
    parser.add_argument('--uniform', type=int, default=0, help='use uniform architecture (all layers same size)')
    parser.add_argument('--run_name', type=str, default='noname')
    parser.add_argument('--disable_hess', type=int, default=1, help='disable hessian because of sysmqrt slowness')
    parser.add_argument('--launch_blocking', type=int, default=0)

    u.seed_random(1)
    gl.args = parser.parse_args()
    args = gl.args
    gl.hacks_disable_hess = args.disable_hess
    u.seed_random(1)

    gl.project_name = 'train_ciresan'
    u.setup_logdir(args.run_name)
    print(f"Logging to {gl.logdir}")

    d1 = 28*28
    if args.uniform:
        d = [784, 784, 784, 784, 784, 784, 10]
    else:
        d = [784, 2500, 2000, 1500, 1000, 500, 10]
    o = 10

    model = u.SimpleFullyConnected2(d, nonlin=args.nonlin, bias=args.bias, dropout=args.dropout)
    model = model.to(gl.device)
    autograd_lib.register(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, original_targets=True,
                          dataset_size=args.dataset_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    assert not args.full_batch, "fixme: validation still uses stats_iter"
    if not args.full_batch:
        stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
        stats_iter = u.infinite_iter(stats_loader)
    else:
        stats_iter = None

    test_dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, train=False, original_targets=True,
                               dataset_size=args.dataset_size)
    test_eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=False)
    train_eval_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=False)


    loss_fn = torch.nn.CrossEntropyLoss()
    autograd_lib.add_hooks(model)
    autograd_lib.disable_hooks()

    gl.token_count = 0
    last_outer = 0

    layer_map = {model.layers[i]: i for i in range(len(model.layers))}

    for step in range(args.stats_steps):
        epoch = gl.token_count // 60000
        print('token_count', gl.token_count)
        if last_outer:
            u.log_scalars({"time/outer": 1000*(time.perf_counter() - last_outer)})
            print(f'time: {time.perf_counter() - last_outer:.2f}')
        last_outer = time.perf_counter()

        with u.timeit("validate"):
            val_accuracy, val_loss = validate(model, test_eval_loader, f'test (epoch {epoch})')
            train_accuracy, train_loss = validate(model, train_eval_loader, f'train (epoch {epoch})')

        # save log
        metrics = {'epoch': epoch, 'val_accuracy': val_accuracy, 'val_loss': val_loss,
                   'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'lr': optimizer.param_groups[0]['lr'],
                   'momentum': optimizer.param_groups[0].get('momentum', 0)}
        u.log_scalars(metrics)

        if not args.skip_stats:
            n = args.stats_batch_size * args.stats_num_batches
            for i in range(args.stats_num_batches):
                activations = {}
                def save_activations(layer, a, _):
                    activations[layer] = a

                print('forward')
                with u.timeit("stats_forward"):
                    with autograd_lib.module_hook(save_activations):
                        data, targets = next(stats_iter)
                        output = model(data)
                        loss = loss_fn(output, targets)

                hessian = defaultdict(lambda: AttrDefault(float))
                fisher = defaultdict(lambda: AttrDefault(float))
                jacobian = defaultdict(lambda: AttrDefault(float))
                current_stats = None

                def compute_stats(layer, _, B):
                    A = activations[layer]
                    # about 27ms per layer
                    with u.timeit('compute_stats'):
                        start = time.time()
                        current_stats[layer].AA += torch.einsum("ni,nj->ij", A, A)
                        current_stats[layer].BB += torch.einsum("ni,nj->ij", B, B)  # TODO(y): index consistency
                        current_stats[layer].diag += torch.einsum("ni,nj->ij", B * B, A * A)
                        current_stats[layer].BA += torch.einsum("ni,nj->ij", B, A)
                        current_stats[layer].norm2 += ((A*A).sum(dim=1) * (B*B).sum(dim=1)).sum()
                        current_stats[layer].n += len(A)

                # todo(y): add "compute_fisher" and "compute_jacobian"
                # todo(y): add couple of statistics (effective rank, trace, gradient noise)
                # todo(y): change to stochastic
                # todo(y): plot eigenvalue spectrum for each, OpenAI and Jain stats, gradient noise
                # todo(y): get convolution working
                # todo(y): test on LeNet5
                # todo(y): test on resnet50
                # todo(y): video of spectra changing

                print('backward')
                with u.timeit("backprop_H"):
                    with autograd_lib.module_hook(compute_stats):
                        current_stats = hessian
                        autograd_lib.backward_hessian(output, loss='CrossEntropy', retain_graph=True)    # 600 ms
                        current_stats = jacobian
                        autograd_lib.backward_jacobian(output, retain_graph=True)   # 600 ms
                        current_stats = fisher
                        model.zero_grad()
                        loss.backward()  # 60 ms

            print('summarize')
            for (i, layer) in enumerate(model.layers):
                stats_dict = {'hessian': hessian, 'jacobian': jacobian, 'fisher': fisher}

                # evaluate stats from
                # https://app.wandb.ai/yaroslavvb/train_ciresan/runs/425pu650?workspace=user-yaroslavvb
                for stats_name in stats_dict:
                    s = AttrDict()
                    stats = stats_dict[stats_name][layer]

                    diag: torch.Tensor = stats.diag / stats.n
                    if stats_name != 'fisher':
                        diag /= o   # extra factor to make diag_trace match kfac version

                    s.diag_l2 = torch.max(diag)     # 40 - 3000 smaller than kfac l2 for jac
                    s.diag_fro = torch.norm(diag)   # jacobian grows to 0.5-1.5, rest falls, layer-5 has phase transition, layer-4 also
                    s.diag_trace = diag.sum()      # jacobian grows 0-1000 (first), 0-150 (last). Almost same as kfac_trace (771 vs 810 kfac). Jacobian has up/down phase transition
                    s.diag_erank = s.diag_trace/torch.max(diag)   # kind of useless, very large and noise, but layer2/jacobian has up/down phase transition

                    # normalize for mean loss
                    BB = stats.BB / stats.n
                    AA = stats.AA / stats.n
                    if stats_name != 'fisher':
                        AA /= o   # jacobian and hessian matrices need another factor of o normalization on a factor
                        A_evals, _ = torch.symeig(AA)   # averaging 120ms per hit, 90 hits
                    B_evals, _ = torch.symeig(BB)
                    s.kfac_l2 = torch.max(A_evals) * torch.max(B_evals)    # 60x larger than diag_l2. layer0/hess has down/up phase transition. layer5/jacobian has up/down phase transition
                    s.kfac_trace = torch.sum(A_evals) * torch.sum(B_evals)  # 0/hess down/up tr, 5/jac sharp phase transition
                    s.kfac_fro = torch.norm(stats.AA) * torch.norm(stats.BB)  # 0/hess has down/up tr, 5/jac up/down transition
                    s.kfac_erank = s.kfac_trace / s.kfac_l2   # first layer has 25, rest 15, all layers go down except last, last noisy

                    s.diversity = (stats.norm2 / n) / u.norm_squared(stats.BA / n)  # gradient diversity. Goes up 3x. Bottom layer has most diversity

                    # discrepancy of KFAC based on exact values of diagonal approximation
                    # average difference normalized by average diagonal magnitude
                    diag_kfac = torch.einsum('ll,ii->li', BB, AA)
                    s.kfac_error = (torch.abs(diag_kfac-diag)).mean()/torch.mean(diag.abs())
                    u.log_scalars(u.nest_stats(f'layer-{i}/{stats_name}', s))

                # openai batch size stat
                s = AttrDict()
                hess = hessian[layer]
                jac = jacobian[layer]
                fish = fisher[layer]
                # the following check passes, but is expensive
                # if args.stats_num_batches == 1:
                #    u.check_close(fisher[layer].BA, layer.weight.grad)

                def trsum(A, B): return (A*B).sum()  # computes tr(AB')
                grad = fisher[layer].BA

                s.hess_curv = trsum(hess.BB/n @ grad, grad @ hess.AA / n / o)  #(hess.BB / n) @ grad @ (hess.AA / n / o)
                s.jac_curv = trsum(jac.BB/n @ grad, grad @ jac.AA / n / o)   # jac.BB / n) @ grad @ (jac.AA / n / o)

                # compute gradient noise statistics
                # fish.BB has /n factor twice, hence don't need extra /n on fish.AA
                s.hess_noise = (trsum(hess.AA / n / o, fish.AA) * trsum(hess.BB / n, fish.BB))
                s.jac_noise = (trsum(jac.AA / n / o, fish.AA) * trsum(jac.BB / n, fish.BB))
                s.hess_noise_normalized = s.hess_noise / (hess.diag.sum() / n)
                s.jac_noise_normalized = s.jac_noise / (jac.diag.sum() / n)
                u.log_scalars(u.nest_stats(f'layer-{i}', s))
                # step size stat
                # rho?

                # TODO(y): check mean error again, check hess_noise, jac_noise



        print('train')
        model.train()
        last_inner = 0
        for i in range(args.train_steps):
            if last_inner:
                u.log_scalars({"time/inner": 1000*(time.perf_counter() - last_inner)})
            last_inner = time.perf_counter()

            optimizer.zero_grad()
            data, targets = next(train_iter)
            model.zero_grad()
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()

            optimizer.step()
            if args.weight_decay:
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.data.mul_(1-args.weight_decay)

            gl.token_count += data.shape[0]

    gl.event_writer.close()


def validate(model, val_loader, tag='validation'):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:

            data, target = data.to(gl.device), target.to(gl.device)

            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    # TODO(y) log scalar here
    print(f'Eval: Average {tag} loss: {val_loss:.4f}, Accuracy: {correct:.0f}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)')

    return val_accuracy, val_loss


if __name__ == '__main__':
    main()

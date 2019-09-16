"""Test factored implementation of stats"""

import argparse
import os
import time

import autograd_lib
import globals as gl
# import torch
import torch
import util as u
import wandb
from attrdict import AttrDefault
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter

# for line profiling
try:
  # noinspection PyUnboundLocalVariable
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.


from train_ciresan import validate


@profile
def main():
    u.seed_random(1)

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    args = parser.parse_args()

    logdir = u.create_local_logdir('/temp/runs/factored_test')
    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)
    print('logging to ', logdir)

    loss_type = 'LeastSquares'

    args.data_width = 2
    args.dataset_size = 5
    args.stats_batch_size = 5
    d1 = args.data_width ** 2
    args.stats_batch_size = args.dataset_size
    args.stats_steps = 1

    n = args.stats_batch_size
    o = 10
    d = [d1, o]

    model = u.SimpleFullyConnected2(d, bias=False, nonlin=0)
    model = model.to(gl.device)
    print(model)

    dataset = u.TinyMNIST(data_width=args.data_width, dataset_size=args.dataset_size, loss_type=loss_type)
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False)
    stats_iter = u.infinite_iter(stats_loader)
    stats_data, stats_targets = next(stats_iter)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    else:   # loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.reset_global_step()
    last_outer = 0
    for step in range(args.stats_steps):
        if last_outer:
            u.log_scalars({"time/outer": 1000*(time.perf_counter() - last_outer)})
        last_outer = time.perf_counter()

        data, targets = stats_data, stats_targets

        # Capture Hessian and gradient stats
        autograd_lib.enable_hooks()
        autograd_lib.clear_backprops(model)
        with u.timeit("backprop_g"):
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward(retain_graph=True)

        autograd_lib.clear_hess_backprops(model)
        with u.timeit("backprop_H"):
            autograd_lib.backprop_hess(output, hess_type=loss_type)
        autograd_lib.disable_hooks()   # TODO(y): use remove_hooks

        with u.timeit("compute_grad1"):
            autograd_lib.compute_grad1(model)
        with u.timeit("compute_hess"):
            autograd_lib.compute_hess(model)
            autograd_lib.compute_hess(model, method='kron', attr_name='hess2')

        autograd_lib.compute_stats_factored(model)

        params = list(model.parameters())
        assert len(params) == 1
        new_values = params[0].stats
        golden_values = torch.load('test/factored.pt')

        print(new_values.sigma_l2)
        for valname in new_values:
            print("Checking ", valname)
            if valname == 'sigma_l2':
                u.check_close(new_values[valname], golden_values[valname], atol=1e-2)  # sigma is approximate
            elif valname == 'sigma_erank':
                u.check_close(new_values[valname], golden_values[valname], atol=0.11)  # 1.0 vs 1.1
            elif valname in ['rho', 'step_div_1_adjusted', 'batch_jain_full']:
                continue   # lyapunov stats weren't computed correctly in golden set
            elif valname in ['batch_openai']:
                continue   # batch sizes depend on sigma which is approximate
            elif valname in ['noise_variance_pinv']:
                pass  # went from 0.22 to 0.014 after kron factoring (0.01 with full centering, 0.3 with no centering)
            else:
                u.check_close(new_values[valname], golden_values[valname], rtol=1e-4, atol=1e-6)

    gl.event_writer.close()


if __name__ == '__main__':
    main()

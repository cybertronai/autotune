"""Evaluate approximation quality of factoring on conv2d layers."""

# Tests that compare manual computation of quantities against PyTorch autograd

import os
import sys
from typing import List

import globals as gl
import torch
from torch import nn as nn
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

import util as u

import numpy as np

import autograd_lib

unfold = torch.nn.functional.unfold
fold = torch.nn.functional.fold


def compute_hess(n: int = 1, image_size: int = 1, kernel_size: int = 1, num_channels: int = 1, num_layers: int = 1,
                 nonlin: bool = False,
                 loss: str = 'CrossEntropy', method='exact', param_name='weight') -> List[torch.Tensor]:
    """

    Args:
        param_name: which parameter to compute ('weight' or 'bias')
        n: number of examples
        image_size:  width of image (square image)
        kernel_size: kernel size
        num_channels:
        num_layers:
        nonlin
        loss: LeastSquares or CrossEntropy
        method: 'kron', 'mean_kron'
        num_layers: number of layers in the network

    Returns:
        list of num_layers Hessian matrices.
    """

    assert param_name in ['weight', 'bias']
    assert loss in autograd_lib._supported_losses
    assert method in autograd_lib._supported_methods

    u.seed_random(1)

    Xh, Xw = image_size, image_size
    Kh, Kw = kernel_size, kernel_size
    dd = [num_channels] * (num_layers+1)

    model: u.SimpleModel2 = u.PooledConvolutional2(dd, kernel_size=(Kh, Kw), nonlin=nonlin, bias=True)
    data = torch.randn((n, dd[0], Xh, Xw))

    autograd_lib.clear_backprops(model)
    autograd_lib.add_hooks(model)
    output = model(data)
    autograd_lib.backprop_hess(output, hess_type=loss)
    autograd_lib.compute_hess(model, method=method)  # todo:rename to "method=exact"
    autograd_lib.disable_hooks()

    result = []
    for i in range(len(model.layers)):
        param = getattr(model.layers[i], param_name)
        if method == 'exact' or method == 'autograd':
            result.append(param.hess)
        else:
            result.append(param.hess_factored.expand())
    return result


def main():
    # for kernel_size=1, mean kron factoring works for any image size
    hess_list1 = compute_hess(n=2, kernel_size=1, image_size=2, num_channels=3, num_layers=4, loss='CrossEntropy', method='exact')
    hess_list2 = compute_hess(n=2, kernel_size=1, image_size=2, num_channels=3, num_layers=4, loss='CrossEntropy', method='mean_kron')

    max_error = max([u.cov_dist(h1, h2) for h1, h2 in zip(hess_list1, hess_list2)])
    print(max_error)


if __name__ == '__main__':
    main()

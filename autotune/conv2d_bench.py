"""Evaluate approximation quality of factoring on conv2d layers.

0.0009374995715916157
tensor(1.0019)
mean_kron
image_size
   value: : [7.420718475259491e-07, 1.611936113476986e-06, 8.632230219518533e-07, 1.5533388477706467e-06]
   magnitude : 1.00, 1.00, 1.00, 1.00
num_channels
   value: : [0.0, 3.0523662530868023e-07, 1.611936113476986e-06, 3.660633751678688e-07]
   magnitude : nan, 1.00, 1.00, 1.00
kernel_size
   value: : [1.611936113476986e-06, 0.028322825208306313, 0.0022797712590545416]
   magnitude : 1.00, 1.03, 1.00
kron
image_size
   value: : [0.0028077568858861923, 0.0009374995715916157, 0.001000001560896635, 0.0008735664887353778]
   magnitude : 1.02, 1.00, 1.00, 1.00
num_channels
   value: : [0.0, 0.0003620386414695531, 0.0009374995715916157, 0.0012453986564651132]
   magnitude : nan, 1.01, 1.00, 1.00
kernel_size
   value: : [0.0009374995715916157, 0.02830260619521141, 0.002279785694554448]
   magnitude : 1.00, 1.03, 1.00

"""

# Tests that compare manual computation of quantities against PyTorch autograd

import os
import sys
from typing import List

import globals as gl
import torch
from attrdict import AttrDict
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
    dd = [num_channels] * (num_layers + 1)

    model: u.SimpleModel2 = u.PooledConvolutional2(dd, kernel_size=(Kh, Kw), nonlin=nonlin, bias=True)
    data = torch.randn((n, dd[0], Xh, Xw))

    autograd_lib.clear_backprops(model)
    autograd_lib.add_hooks(model)
    output = model(data)
    autograd_lib.backprop_hess(output, hess_type=loss)
    autograd_lib.compute_hess(model, method=method)
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
    main_vals = AttrDict(n=2, kernel_size=1, image_size=2, num_channels=3, num_layers=4, loss='CrossEntropy',
                    nonlin=False)

    hess_list1 = compute_hess(method='exact', **main_vals)
    hess_list2 = compute_hess(method='kron', **main_vals)
    value_error = max([u.cov_dist(h1, h2) for h1, h2 in zip(hess_list1, hess_list2)])
    magnitude_error = max([u.l2_norm(h2) / u.l2_norm(h1) for h1, h2 in zip(hess_list1, hess_list2)])
    print(value_error)
    print(magnitude_error)

    for method in ['mean_kron', 'kron']:
        print(method)
        for dimension in ['image_size', 'num_channels', 'kernel_size']:
            value_errors = []
            magnitude_errors = []
            for i in range(1, 5):
                if dimension == 'kernel_size' and i > 3:
                    break
                vals = AttrDict(main_vals.copy())
                vals.method = method
                vals[dimension] = i
                vals.image_size = max(vals.image_size, vals.kernel_size ** vals.num_layers)
                # print(vals)
                vals_exact = AttrDict(vals.copy())
                vals_exact.method = 'exact'
                hess_list1 = compute_hess(**vals_exact)
                hess_list2 = compute_hess(**vals)
                value_error = max([u.cov_dist(h1, h2) for h1, h2 in zip(hess_list1, hess_list2)])
                magnitude_error = max([u.l2_norm(h2) / u.l2_norm(h1) for h1, h2 in zip(hess_list1, hess_list2)])
                value_errors.append(value_error)
                magnitude_errors.append(magnitude_error.item())
            print(dimension)
            print('   value: :', value_errors)
            print('   magnitude :', u.format_list(magnitude_errors))


if __name__ == '__main__':
    main()

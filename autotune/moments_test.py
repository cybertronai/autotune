# self-contained example computing all the quantities to reproduce learning rates for (1,2) Gaussian and toy example
# https://www.wolframcloud.com/obj/yaroslavvb/newton/autotune_unit_tests.nb

import sys
import os
import sys

import globals as gl
import pytest
import torch
from torch import nn as nn
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import util as u
import numpy as np
import autograd_lib
import math
import scipy


class PairedLookup(nn.Module):
    """Neural network simulating given set of As/Bs"""
    def __init__(self, As, Bs):
        super().__init__()
        self.As = u.from_numpy(As).float()
        self.Bs = u.from_numpy(Bs).float()
        (n, d1) = self.As.shape
        (_, d2) = self.Bs.shape
        self.layer = torch.nn.Linear(d1, d2, bias=False)
        self.layer.weight.data.copy_(torch.eye(d1))
        assert d1 == d2
        self.d1 = d1

        self.to(gl.device)

    def forward(self, batch: torch.Tensor):
        def find_index(dictionary, entry):
          for (i, e) in enumerate(dictionary):
            if (e == entry).all():
              return i
          return -1

        outputs = []
        assert u.is_matrix(batch)
        batch = self.layer(batch)   # ensure hooks are registered
        for x in batch:
          i = find_index(self.As, x)
          assert i >= 0
          scalar = torch.dot(x, self.Bs[i])
          outputs.append(scalar.reshape((1, 1)))  # backward_jacobians assumes batch+multiclass
        return torch.cat(outputs)


def reconstruct_observations(dataset, data_width, batch_size):
    """Reconstructs matrix of observations by propagating given dataset on identity layer"""

    torch.set_default_dtype(torch.float32)

    model = u.SimpleFullyConnected2([data_width, data_width])
    autograd_lib.register(model)

    model.layers[0].weight.data.copy_(torch.eye(data_width))
    activations = {}

    assert len(dataset) % batch_size == 0
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    X = []

    for data, label in train_loader:
        assert data.shape == (batch_size, data_width
                              )
        def save_activations(layer, A, _):
            activations[layer] = A
        with autograd_lib.module_hook(save_activations):
            output = model(data)
            fake_target = output.detach() - label
            loss = u.least_squares(output, fake_target, aggregation='sum')

        def accumulate_dataset(layer, _, B):
            A = activations[layer]

            n = len(B)
            assert n == len(A)

            x = torch.einsum('ni,nj->nij', A, B).reshape(n, -1)
            if n == 1:  # alternative representations
                x2 = u.c2v(u.vec(u.outer(B[0], A[0])))  # vec(BA')
                u.check_equal(x.flatten(), x2)
                x3 = u.kron(A[0], B[0])  # A\otimes B
                u.check_equal(x.flatten(), x3)
            X.append(x)

        with autograd_lib.module_hook(accumulate_dataset):
            loss.backward()

    return torch.cat(X)


def test_toy_dataset():
    torch.set_default_dtype(torch.float32)

    # https://www.wolframcloud.com/obj/yaroslavvb/newton/autotune_unit_tests.nb
    X0 = [[3, -1, 9, -3], [2, -4, 1, -2], [1, -1, 1, -1]]
    dataset = u.ToyDataset()
    X = reconstruct_observations(dataset, 2, 1)
    u.check_equal(X, X0)
    X = reconstruct_observations(dataset, 2, 3)
    u.check_equal(X, X0)


def _setup_toy_model():
    torch.set_default_dtype(torch.float32)
    As, Bs = [[[1, 3], [2, 1], [-1, -1]], [[3, -1], [1, -2], [-1, 1]]]
    As = u.from_numpy(As).float()
    Bs = u.from_numpy(Bs).float()
    model = PairedLookup(As, Bs)
    autograd_lib.register(model)
    return As, Bs, model


def test_toy_jacobian():
    """Test Jacobian propagation for toy problem."""
    As, Bs, model = _setup_toy_model()
    d = 2
    AA = [torch.zeros(d, d)]
    BB = [torch.zeros(d, d)]
    ABAB = [torch.zeros(d, d, d, d)]

    batch_size = 1
    dataset = u.ToyDataset()
    assert len(dataset) % batch_size == 0
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    activations = {}
    for a in As:
        data = u.from_numpy(a)
        data = data.unsqueeze(0)   # add batch dimension
        def save_activations(layer, A, _): activations[layer] = A
        with autograd_lib.module_hook(save_activations):
            output = model(data)

        def save_backprops(layer, _, B):
            A = activations[layer]
            n = A.shape[0]
            AA[0] += torch.einsum("ni,nj->ij", A, A)
            BB[0] += torch.einsum("ni,nj->ij", B, B)
            xs = torch.einsum("ni,nj->nij", A, B).reshape(n, -1)
            ABAB[0] += torch.einsum("ni,nj,nk,nl->ijkl", A, B, A, B)
            print(xs)
        with autograd_lib.module_hook(save_backprops):
            autograd_lib.backward_jacobian(output)

    n = len(As)
    ABAB[0] /= n
    AA[0] /= n
    BB[0] /= n
    # XX'
    truth = [[14/3, -4, 10, -(14/3)], [-4, 6, -(14/3), 4], [10, -(14/3), 83/  3, -10], [-(14/3), 4, -10, 14/3]]
    u.check_equal(ABAB[0].reshape(d**2, d**2), truth)

    # AA'\otimes BB'
    truth = [[22/3, -4, 22/3, -4], [-4, 4, -4, 4], [22/3, -4, 121/9, -(22/3)], [-4,   4, -(22/3), 22/3]]
    u.check_equal(u.kron(AA[0], BB[0]), truth)


def test_offline_toy():
    """Offline estimation of stochastic alpha and deterministic alpha"""

    As, Bs, model = _setup_toy_model()
    d = 2
    AA = [torch.zeros(d, d)]
    BB = [torch.zeros(d, d)]
    ABAB = [torch.zeros(d, d, d, d)]

    activations = {}
    for a in As:
        data = u.from_numpy(a)
        data = data.unsqueeze(0)   # add batch dimension
        def save_activations(layer, A, _): activations[layer] = A
        with autograd_lib.module_hook(save_activations):
            output = model(data)

        def save_backprops(layer, _, B):
            A = activations[layer]
            n = A.shape[0]
            AA[0] += torch.einsum("ni,nj->ij", A, A)
            BB[0] += torch.einsum("ni,nj->ij", B, B)
            xs = torch.einsum("ni,nj->nij", A, B).reshape(n, -1)
            ABAB[0] += torch.einsum("ni,nj,nk,nl->ijkl", A, B, A, B)
            print(xs)
        with autograd_lib.module_hook(save_backprops):
            autograd_lib.backward_jacobian(output)

    n = len(As)
    ABAB[0] /= n



def main():
    u.run_all_tests(sys.modules[__name__])


if __name__ == '__main__':
    main()

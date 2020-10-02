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
    return  u.ToyDataset(), model


def test_toy_jacobian():
    """Test Jacobian propagation for toy problem."""
    dataset, model = _setup_toy_model()
    d = 2
    AA = [torch.zeros(d, d)]
    BB = [torch.zeros(d, d)]
    X2 = [torch.zeros(d, d, d, d)]
    X2X2 = [torch.zeros(d, d, d, d)]

    batch_size = 1
    assert len(dataset) % batch_size == 0
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    activations = {}
    n = 0
    for data, _ in train_loader:
        n += len(data)
        def save_activations(layer, A, _): activations[layer] = A
        with autograd_lib.module_hook(save_activations):
            output = model(data)

        def save_backprops(layer, _, B):
            A = activations[layer]
            AA[0] += torch.einsum("ni,nj->ij", A, A)
            BB[0] += torch.einsum("ni,nj->ij", B, B)
            X2[0] += torch.einsum("ni,nj,nk,nl->ijkl", A, B, A, B)
            X2X2[0] += torch.einsum("ni,nj,nk,nl,nk,nl,np,nq->ijpq", A, B, A, B, A, B, A, B)
        with autograd_lib.module_hook(save_backprops):
            autograd_lib.backward_jacobian(output)

    X2 = X2[0] / n
    X2X2 = X2X2[0] / n
    AA = AA[0] / n
    BB = BB[0] / n
    X2 = X2.reshape(d**2, d**2)
    X2X2 = X2X2.reshape(d ** 2, d ** 2)

    # XX'
    truth = [[14/3, -4, 10, -(14/3)], [-4, 6, -(14/3), 4], [10, -(14/3), 83/3, -10], [-(14/3), 4, -10, 14/3]]
    u.check_equal(X2, truth)

    # AA'\otimes BB'
    truth = [[22/3, -4, 22/3, -4], [-4, 4, -4, 4], [22/3, -4, 121/9, -(22/3)], [-4,   4, -(22/3), 22/3]]
    u.check_equal(u.kron(AA, BB), truth)

    truth = [[1004/3, -168, 918, -(1004/3)], [-168, 168, -(1004/3),   168], [918, -(1004/3), 8129/3, -918], [-(1004/3), 168, -918, 1004/3]]
    u.check_equal(X2X2, truth)

    rho_sto = u.spectral_radius_real(X2X2@u.pinv(2*X2))
    u.check_close(1/rho_sto, 0.02)

    rho_det = u.spectral_radius_real(X2)/2
    u.check_close(1/rho_det, 2*0.0271278)


def test_gauss12_offline():
    """Gaussian 1, 2"""

    d = 2
    AA = [torch.zeros(d, d)]
    BB = [torch.zeros(d, d)]
    X2 = [torch.zeros(d, 1, d, 1)]
    X2X2 = [torch.zeros(d, 1, d, 1)]

    num_steps = 10
    batch_size = 100000

    # todo: add random rotation
    sigma = torch.diag(torch.tensor([1., 2]))
    u.seed_random(1)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), sigma)
    model = u.SimpleFullyConnected2([2, 1])
    autograd_lib.register(model)
    activations = {}

    n = 0
    for i in range(num_steps):
        data = m.sample([batch_size])

        n += len(data)
        def save_activations(layer, A, _): activations[layer] = A
        with autograd_lib.module_hook(save_activations):
            output = model(data)

        def save_backprops(layer, _, B):
            A = activations[layer]
            AA[0] += torch.einsum("ni,nj->ij", A, A)
            BB[0] += torch.einsum("ni,nj->ij", B, B)
            X2[0] += torch.einsum("ni,nj,nk,nl->ijkl", A, B, A, B)
            X2X2[0] += torch.einsum("ni,nj,nk,nl,nk,nl,np,nq->ijpq", A, B, A, B, A, B, A, B)
        with autograd_lib.module_hook(save_backprops):
            autograd_lib.backward_jacobian(output)

    X2 = X2[0] / n
    X2 = X2.reshape(d, d)
    u.check_close(X2, [[1, 0], [0, 2]], atol=0.004)
    X2X2 = X2X2[0] / n
    X2X2 = X2X2.reshape([d, d])
    u.check_close(X2X2, [[5, 0], [0, 14]], atol=0.1)
    AA = AA[0] / n
    BB = BB[0] / n

    rho_sto = u.spectral_radius_real(X2X2@u.pinv(2*X2))
    u.check_close(1/rho_sto, 1/3.5)

    rho_det = u.spectral_radius_real(X2)/2
    u.check_close(1/rho_det, 1, atol=0.01)


def main():
    u.run_all_tests(sys.modules[__name__])


if __name__ == '__main__':
    main()

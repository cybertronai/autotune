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
    BB = [torch.zeros(1, 1)]
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

    u.check_close(u.kron(AA, BB), X2, atol=0.004)

    rho_sto = u.spectral_radius_real(X2X2@u.pinv(2*X2))
    u.check_close(1/rho_sto, 1/3.5)

    rho_det = u.spectral_radius_real(X2)/2
    u.check_close(1/rho_det, 1, atol=0.01)


def test_gauss12_online():
    """Gaussian 1, 2"""

    d = 2
    AA = [torch.zeros(d, d)]
    BB = [torch.zeros(1, 1)]
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

    u.check_close(u.kron(AA, BB), X2, atol=0.004)

    rho_sto = u.spectral_radius_real(X2X2@u.pinv(2*X2))
    u.check_close(1/rho_sto, 1/3.5)

    rho_det = u.spectral_radius_real(X2)/2
    u.check_close(1/rho_det, 1, atol=0.01)


from collections import namedtuple, defaultdict

import autograd_lib
import pytest

import util as u

# Test exact Hessian computation

# import torch
from typing import Callable

import torch
import torch.nn as nn

from attrdict import AttrDefault, AttrDict

import torch.utils.data as data


def simple_model(d, num_layers):
    """Creates simple linear neural network initialized to identity"""
    layers = []
    for i in range(num_layers):
        layer = nn.Linear(d, d, bias=False)
        layer.weight.data.copy_(torch.eye(d))
        layers.append(layer)
    return torch.nn.Sequential(*layers)


class GaussianDataset(data.Dataset):
    def __init__(self, d):
        super().__init__()

        self.d = d
        sigma = torch.diag(torch.arange(1, 2 * d + 1).float())
        mu = torch.ones(2 * d)
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
        self.d = d

    def __getitem__(self, index):
        sample = self.dist.sample()
        return sample.split(self.d)

    def __len__(self):
        return sys.maxsize



def test_wicks():

    u.seed_random(1)
    d = 1
    n = 10000   # total number of examples
    batch_size = n
    num_processed = 0

    dataset = GaussianDataset(d)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = simple_model(d, 1)
    autograd_lib.register(model)

    activations = {}
    moments = defaultdict(lambda: AttrDefault(float))

    def save_activations(layer, a, _):
      activations[layer] = a

    for data, label in train_loader:
      with autograd_lib.module_hook(save_activations):
        output = model.forward(data)

      def compute_hess(layer, _, B):
        A = activations[layer]
        moments[layer].a += torch.einsum("ni->i", A)
        moments[layer].b += torch.einsum("nj->j", B)
        moments[layer].AA += torch.einsum('ni,nk->ik', A, A)
        moments[layer].AB += torch.einsum("ni,nj->ij", A, B)
        moments[layer].BA += torch.einsum("nj,ni->ji", B, A)
        moments[layer].BB += torch.einsum("nj,nl->jl", B, B)
        moments[layer].BABA += torch.einsum('nl,nk,nj,ni->lkji', B, A, B, A)

      with autograd_lib.module_hook(compute_hess):
        output.backward(label)

      num_processed += len(data)
      if num_processed >= n:
        break

    m = moments[model[0]]
    m = u.divide_attributes(m, n)
    u.check_equal(torch.round(m.BABA), [[6]])
    # m.AA * m.BB
    u.check_equal(torch.round(m.AA*m.BB+m.AB*m.BA+m.BA*m.AB-2*m.a*m.a*m.b*m.b), [[6]])


class IntegerDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.x = torch.tensor([[-12, 0, -8, -4], [12, 12, 12, 0], [-4, 12, 12, -4], [4, 0, 4, 0]]).float()

    def __getitem__(self, index):
        return self.x[index].split(2)

    def __len__(self):
        return 4


def accumulate_moments(train_loader, model, n=0):
    """Feeds data from given loader through the model, captures all backprop and activation moments."""

    activations = {}
    moments = defaultdict(lambda: AttrDefault(float))

    num_processed = 0
    def save_activations(layer, a, _):
      activations[layer] = a

    for data, label in train_loader:
      with autograd_lib.module_hook(save_activations):
        output = model.forward(data)

      def compute_hess(layer, _, B):
        A = activations[layer]
        moments[layer].a += torch.einsum("ni->i", A)
        moments[layer].b += torch.einsum("nj->j", B)
        moments[layer].AA += torch.einsum('ni,nk->ik', A, A)
        moments[layer].AB += torch.einsum("ni,nj->ij", A, B)
        moments[layer].BA += torch.einsum("nj,ni->ji", B, A)
        moments[layer].BB += torch.einsum("nj,nl->jl", B, B)
        moments[layer].BABA += torch.einsum('nl,nk,nj,ni->lkji', B, A, B, A)

      with autograd_lib.module_hook(compute_hess):
        output.backward(label)

      num_processed += len(data)
      if n and num_processed >= n:
        break

    # normalize
    for layer in moments:
      moments[layer] = u.divide_attributes(moments[layer], num_processed)

    return moments


def test_integer():
    # test forward, backward on exact dataset
    d = 2
    dataset = IntegerDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = simple_model(d, 1)
    autograd_lib.register(model)

    moments = accumulate_moments(train_loader, model)
    m = moments[model[0]]
    u.check_equal(m.BABA, [[[[8128, 3456], [960, 576]], [[3456, 10368], [576, -1728]]],
                           [[[960, 576], [640, -192]], [[576, -1728], [-192, 576]]]])
    u.check_equal(m.BA, [[52, 72], [16, -12]])
    u.check_equal(m.AA, [[80, 24], [24, 72]])
    u.check_equal(m.BB, [[92, -4], [-4, 8]])
    u.check_equal(m.a, [0, 6])
    u.check_equal(m.b, [5, -2])

    X = u.from_numpy([[1, 2], [3, 4]]).float()

    def exact_forward(X):
        return torch.einsum("lkji,ji->lk", m.BABA, X)

    def kfac_forward_indexed(X):
        return torch.einsum('lj,ik,ji->lk', m.BB, m.AA, X)

    def kfac_forward_matrix(X):
        return m.BB @ X @ m.AA

    kfac_forward = kfac_forward_matrix

    def dot(x, y):
        return torch.dot(x.flatten(), y.flatten())

    def wicks_forward(X):
        BA0 = u.outer(m.b, m.a)
        return m.BB @ X @ m.AA + (m.AB @ X @ m.AB).T + m.BA*dot(X, m.BA) - 2*u.outer(m.b, m.a)*dot(BA0, X)

    def kfac_backward(Y):
        return u.pinv(m.BB) @ Y @ u.pinv(m.AA)

    u.check_equal(exact_forward(X), [[20224, 19008], [3264, -1152]])
    u.check_equal(kfac_forward_indexed(X), [[10432, 14016], [2176, 2208]])
    u.check_equal(kfac_forward_matrix(X),  [[10432, 14016], [2176, 2208]])

    u.check_equal(wicks_forward(X), [[37920, 36192], [4896, -432]])

    u.check_close(kfac_backward(kfac_forward(X)), X)





def main():
    pass
    test_integer()
    # u.run_all_tests(sys.modules[__name__])


if __name__ == '__main__':
    main()

# Test exact Hessian computation

import os
import sys

# import torch
import torch
import torch.nn as nn
from torchcurv.optim import SecondOrderOptimizer

import util as u


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result


def test_simple_hessian():
    # Compare against manual calculations in
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/linear-jacobians-and-hessians.nb
    torch.set_default_dtype(torch.float32)

    d = [2, 3, 4, 2]
    n = d[0]
    c = d[-1]
    As = torch.tensor([[3, 1, -1], [1, -3, -2]]).float()
    Bs = torch.tensor([[[3, -3], [-3, -1], [-3, 3], [-3, 0]], [[2, -1], [-3, 0], [1, 1], [-2, 0]]]).float()

    # output Jacobian for first example
    Jo1 = u.kron(u.v2r(As[0]), Bs[0].t())
    u.check_equal(Jo1, [[9, -9, -9, -9, 3, -3, -3, -3, -3, 3, 3, 3], [-9, -3, 9, 0, -3, -1, 3, 0, 3, 1, -3, 0]])

    # batch output Jacobian
    Jb = torch.cat([u.kron(u.v2r(As[i]), Bs[i].t()) for i in range(n)])
    u.check_equal(Jb, [[9, -9, -9, -9, 3, -3, -3, -3, -3, 3, 3, 3], [-9, -3, 9, 0, -3, -1, 3, 0, 3, 1, -3, 0],
                       [2, -3, 1, -2, -6, 9, -3, 6, -4, 6, -2, 4], [-1, 0, 1, 0, 3, 0, -3, 0, 2, 0, -2, 0]])

    W = torch.nn.Parameter(torch.ones((d[2], d[1])))

    def loss(i):
        residuals = Bs[i].t() @ W @ u.v2c(As[i])
        return 0.5 * torch.sum(residuals * residuals)

    u.check_equal(loss(0), 333 / 2)

    # check against PyTorch autograd
    i = 0
    outputs = Bs[i].t() @ W @ u.v2c(As[i])
    jac = u.jacobian(outputs, W)

    u.check_equal(Jo1, jac.transpose(0, 1).transpose(2, 3).reshape((c, -1)))

    Jb = torch.cat([u.kron(u.v2r(As[i]), Bs[i].t()) for i in range(n)])
    manualHess = Jb.t() @ Jb
    u.check_equal(manualHess, [[167, -60, -161, -85, 39, 0, -57, -15, -64, 30, 52, 35],
                               [-60, 99, 51, 87, 0, 3, 27, 9, 30, -48, -12, -39],
                               [-161, 51, 164, 79, -57, 27, 48, 33, 52, -12, -58, -23],
                               [-85, 87, 79, 85, -15, 9, 33, 15, 35, -39, -23, -35],
                               [39, 0, -57, -15, 63, -60, -9, -45, 12, -30, 24, -15],
                               [0, 3, 27, 9, -60, 91, -21, 63, -30, 44, -24, 27],
                               [-57, 27, 48, 33, -9, -21, 36, -9, 24, -24, -6, -21],
                               [-15, 9, 33, 15, -45, 63, -9, 45, -15, 27, -21, 15],
                               [-64, 30, 52, 35, 12, -30, 24, -15, 38, -30, -14, -25],
                               [30, -48, -12, -39, -30, 44, -24, 27, -30, 46, -6, 33],
                               [52, -12, -58, -23, 24, -24, -6, -21, -14, -6, 26, 1],
                               [35, -39, -23, -35, -15, 27, -21, 15, -25, 33, 1, 25]])

    total_loss = torch.add(*[loss(i) for i in range(n)])
    u.check_equal(total_loss, 397 / 2)

    automaticHess = u.hessian(total_loss, W)
    automaticHess = automaticHess.transpose(0, 1).transpose(2, 3).reshape((d[1] * d[2], d[1] * d[2]))
    u.check_equal(automaticHess, manualHess)

    # Note: layers have dimensions (in, out), but the matrices have shape (out, in)
    layer = nn.Linear(d[1], d[2], bias=False)
    Blayer = nn.Linear(d[2], d[3], bias=False)
    model = torch.nn.Sequential(layer, nn.ReLU(), Blayer)
    layer.weight.data.copy_(torch.ones((d[2], d[1])))
    Blayer.weight.data.copy_(Bs[0].t())
    u.check_close(model(As[0]), [-18., -3.])


import argparse
import os
import sys
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


def test_explicit_hessian():
    """Check computation of hessian of loss(B'WA) from https://github.com/yaroslavvb/kfac_pytorch/blob/master/derivation.pdf


    """

    torch.set_default_dtype(torch.float64)
    A = torch.tensor([[-1., 4], [3, 0]])
    B = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    Y = B.t() @ X @ A
    u.check_equal(Y, [[-52, 64], [-81, -108]])
    loss = torch.sum(Y * Y) / 2
    hess0 = u.hessian(loss, X).reshape([4, 4])
    hess1 = u.Kron(A @ A.t(), B @ B.t())

    u.check_equal(loss, 12512.5)

    # PyTorch autograd computes Hessian with respect to row-vectorized parameters, whereas
    # autograd_lib uses math convention and does column-vectorized.
    # Commuting order of Kronecker product switches between two representations
    u.check_equal(hess1.commute(), hess0)

    # Do a test using Linear layers instead of matrix multiplies
    model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([2, 2, 2], bias=False)
    model.layers[0].weight.data.copy_(X)

    # Linear layers are equivalent to multiplying on the right, treating data matrices as rows of datapoints
    # Transpose to match previous results
    u.check_equal(model.layers[0](A.t()).t(), [[5, -20], [-16, -8]])  # XA = (A'X0)'

    model.layers[1].weight.data.copy_(B.t())
    u.check_equal(model(A.t()).t(), Y)

    Y = model(A.t()).t()    # transpose to data-dimension=columns
    loss = torch.sum(Y * Y) / 2
    loss.backward()

    u.check_equal(model.layers[0].weight.grad, [[-2285, -105], [-1490, -1770]])
    G = B @ Y @ A.t()
    u.check_equal(model.layers[0].weight.grad, G)

    u.check_equal(hess0, u.Kron(B @ B.t(), A @ A.t()))

    # compute newton step
    u.check_equal(u.Kron(A@A.t(), B@B.t()).pinv() @ u.vec(G), u.v2c([-5, -2, 0, -6]))

    # compute Newton step using factored representation
    autograd_lib.add_hooks(model)

    Y = model(A.t())
    n = 2
    loss = torch.sum(Y * Y) / 2
    autograd_lib.backprop_hess(Y, hess_type='LeastSquares')
    autograd_lib.compute_hess(model, method='kron', attr_name='hess_kron', vecr_order=False, loss_aggregation='sum')
    param = model.layers[0].weight
    hess2 = param.hess_kron
    print(hess2)

    u.check_equal(hess2, [[425, 170, -75, -30], [170, 680, -30, -120], [-75, -30, 225, 90], [-30, -120, 90, 360]])

    # Gradient test
    model.zero_grad()
    loss.backward()
    u.check_close(u.vec(G).flatten(), u.Vec(param.grad))

    # Newton step test
    # Method 0: PyTorch native autograd
    newton_step0 = param.grad.flatten() @ torch.pinverse(hess0)
    newton_step0 = newton_step0.reshape(param.shape)
    u.check_equal(newton_step0, [[-5, 0], [-2, -6]])

    # Method 1: colummn major order
    ihess2 = hess2.pinv()
    u.check_equal(ihess2.LL, [[1/16, 1/48], [1/48, 17/144]])
    u.check_equal(ihess2.RR, [[2/45, -(1/90)], [-(1/90), 1/36]])
    u.check_equal(torch.flatten(hess2.pinv() @ u.vec(G)), [-5, -2, 0, -6])
    newton_step1 = (ihess2 @ u.Vec(param.grad)).matrix_form()

    # Method2: row major order
    newton_step2 = ihess2.commute() @ u.Vecr(param.grad)
    newton_step2 = newton_step2.matrix_form()

    u.check_equal(newton_step0, newton_step1)
    u.check_equal(newton_step0, newton_step2)


def _test_factored_hessian():
    """"Simple test to ensure Hessian computation is working.

    In a linear neural network with squared loss, Newton step will converge in one step.
    Compute stats after minimizing, pass sanity checks.
    """

    u.seed_random(1)
    loss_type = 'LeastSquares'

    data_width = 2
    n = 5
    d1 = data_width ** 2
    o = 10
    d = [d1, o]

    model = u.SimpleFullyConnected2(d, bias=False, nonlin=False)
    model = model.to(gl.device)
    print(model)

    dataset = u.TinyMNIST(data_width=data_width, dataset_size=n, loss_type=loss_type)
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    stats_iter = u.infinite_iter(stats_loader)
    stats_data, stats_targets = next(stats_iter)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    else:  # loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.reset_global_step()
    last_outer = 0

    data, targets = stats_data, stats_targets

    # Capture Hessian and gradient stats
    autograd_lib.enable_hooks()
    autograd_lib.clear_backprops(model)

    output = model(data)
    loss = loss_fn(output, targets)
    print(loss)
    loss.backward(retain_graph=True)
    layer = model.layers[0]

    autograd_lib.clear_hess_backprops(model)
    autograd_lib.backprop_hess(output, hess_type=loss_type)
    autograd_lib.disable_hooks()

    # compute Hessian using direct method, compare against PyTorch autograd
    hess0 = u.hessian(loss, layer.weight)
    autograd_lib.compute_hess(model)
    hess1 = layer.weight.hess
    print(hess1)
    u.check_close(hess0.reshape(hess1.shape), hess1, atol=1e-9, rtol=1e-6)

    # compute Hessian using factored method
    autograd_lib.compute_hess(model, method='kron', attr_name='hess2', vecr_order=True)
    # s.regret_newton = vecG.t() @ pinvH.commute() @ vecG.t() / 2  # TODO(y): figure out why needed transposes

    hess2 = layer.weight.hess2
    u.check_close(hess1, hess2, atol=1e-9, rtol=1e-6)

    # Newton step in regular notation
    g1 = layer.weight.grad.flatten()
    newton1 = hess1 @ g1

    g2 = u.Vecr(layer.weight.grad)
    newton2 = g2 @ hess2

    u.check_close(newton1, newton2, atol=1e-9, rtol=1e-6)

    # compute regret in factored notation, compare against actual drop in loss
    regret1 = g1 @ hess1.pinverse() @ g1 / 2
    regret2 = g2 @ hess2.pinv() @ g2 / 2
    u.check_close(regret1, regret2)

    current_weight = layer.weight.detach().clone()
    param: torch.nn.Parameter = layer.weight
    # param.data.sub_((hess1.pinverse() @ g1).reshape(param.shape))
    # output = model(data)
    # loss = loss_fn(output, targets)
    # print("result 1", loss)

    # param.data.sub_((hess1.pinverse() @ u.vec(layer.weight.grad)).reshape(param.shape))
    # output = model(data)
    # loss = loss_fn(output, targets)
    # print("result 2", loss)

    # param.data.sub_((u.vec(layer.weight.grad).t() @ hess1.pinverse()).reshape(param.shape))
    # output = model(data)
    # loss = loss_fn(output, targets)
    # print("result 3", loss)
    #

    del layer.weight.grad
    output = model(data)
    loss = loss_fn(output, targets)
    loss.backward()
    param.data.sub_(u.unvec(hess1.pinverse() @ u.vec(layer.weight.grad), layer.weight.shape[0]))
    output = model(data)
    loss = loss_fn(output, targets)
    print("result 4", loss)

    # param.data.sub_((g1 @ hess1.pinverse() @ g1).reshape(param.shape))

    print(loss)


if __name__ == '__main__':
    #  _test_factored_hessian()
    _test_explicit_hessian()
    #    u.run_all_tests(sys.modules[__name__])

    # u.run_all_tests(sys.modules[__name__])

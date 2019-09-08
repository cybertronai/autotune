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

    # Run optimizer to capture A's and B's
    optim_kwargs = dict(lr=0, momentum=0, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", )
    curv_args = dict(damping=1, ema_decay=1)  # todo: damping
    _optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)


if __name__ == '__main__':
    u.run_all_tests(sys.modules[__name__])

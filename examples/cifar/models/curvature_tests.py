# Prototype batch-size quantities from
# https://docs.google.com/document/d/19Jmh4spbSAnAGX_eq7WSFPgLzrpJEhiZRpjX1jSYObo/edit

import numpy as np
import torch
import torch.nn as nn

from torchcurv.optim import SecondOrderOptimizer


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w(x)


def test_loss():
    # Reproduce Linear Regression example
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/curvature-unit-tests.nb

    d = 2
    n = 3
    model = Net(d)
    model.w.weight.data.copy_(torch.from_numpy(np.array([[1, 2]])))

    X = torch.tensor([[-2, 0, 2], [-1, 1, 3]]).float()
    assert X.shape[0] == d
    assert X.shape[1] == n

    Y = torch.tensor([[0, 1, 2]]).float()
    assert Y.shape[1] == X.shape[1]

    w0 = torch.tensor([[1, 2]])
    assert w0.shape[1] == d

    data = X.t()       # PyTorch expects batch dimension first
    target = Y.t()
    assert len(data) == n

    output = model(data)
    residuals = output - Y.t()

    def compute_loss(residuals_):
        return torch.sum(residuals_ * residuals_) / (2 * n)

    loss = compute_loss(residuals)

    print(repr(loss.detach().numpy()))
    assert loss-8.83333 < 1e-5, torch.norm(loss)-8.83333
    print(loss)

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0, curv_type="Fisher", curv_shapes={"Linear": "Kron"}, momentum=0,
                        momentum_type="preconditioned", weight_decay=0)
    curv_args = dict(damping=0, ema_decay=1)
    optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)

    # update params
    def closure_fisher():
        """Closure for estimating empirical Fisher"""
        for group in optimizer.param_groups:
            group['curv'].prob = torch.ones_like(Y)  # q: what is prob for?
            group['curv'].turn_off_backward()
        optimizer.zero_grad()
        output_ = model(data)
        loss_ = compute_loss(output_ - Y.t())
        loss_.backward(create_graph=False)

        return loss_, output_

    num_mc = 1

    def closure_hessian():
        """Closure for estimating Hessian"""
        optimizer.zero_grad()
        output_ = model(data)

        for group in optimizer.param_groups:
            group['curv'].prob = torch.ones_like(Y)
            group['curv'].turn_on_backward()
            for param in group['params']:
                param.requires_grad = False

        for group in optimizer.param_groups:
            group['curv'].prob = torch.ones_like(target)

            for i in range(num_mc):
                loss_ = compute_loss(output_ - target)
                loss_.backward(retain_graph=True)

        for group in optimizer.param_groups:
            group['curv'].turn_off_backward()
            for param in group['params']:
                param.requires_grad = True

    loss = compute_loss(output - Y.t())
    loss.backward()

    loss, output = optimizer.step(closure=closure_fisher())
    # TODO: get A's and B's, compute Sigma, Sigma_c, gradient diversity,

    loss, output = optimizer.step(closure=closure_hessian())
    # TODO: get A's and B's, compute H, rho, Newton decrement, all learning rate + batch size stats
    print(loss.item())



if __name__ == '__main__':
    test_loss()

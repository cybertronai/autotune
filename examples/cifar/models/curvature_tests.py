# Prototype batch-size quantities from
# Batch size formulas (https://docs.google.com/document/d/19Jmh4spbSAnAGX_eq7WSFPgLzrpJEhiZRpjX1jSYObo/edit)
from typing import Optional, Tuple, Callable

import scipy
import torch
import torch.nn as nn
from torchcurv.optim import SecondOrderOptimizer

import numpy as np


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result


def test_loss():
    # Reproduce Linear Regression example
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/curvature-unit-tests.nb

    d = 2
    n = 3
    model = Net(d)

    w0 = torch.tensor([[1, 2]]).float()
    assert w0.shape[1] == d
    model.w.weight.data.copy_(w0)

    X = torch.tensor([[-2, 0, 2], [-1, 1, 3]]).float()
    assert X.shape[0] == d
    assert X.shape[1] == n

    Y = torch.tensor([[0, 1, 2]]).float()
    assert Y.shape[1] == X.shape[1]

    data = X.t()  # PyTorch expects batch dimension first
    target = Y.t()
    assert data.shape[0] == n

    output = model(data)
    # residuals, aka e
    residuals = output - Y.t()

    def compute_loss(residuals_):
        return torch.sum(residuals_ * residuals_) / (2 * n)

    loss = compute_loss(residuals)

    print(repr(loss.detach().numpy()))
    assert loss - 8.83333 < 1e-5, torch.norm(loss) - 8.83333
    print(loss)

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0, momentum=0, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", )
    curv_args = dict(damping=1, ema_decay=1)  # todo: damping
    optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)

    # def set_requires_grad(v):
    #     for p in model.parameters():
    #         p.requires_grad = False
    #
    def backward(last_layer: str) -> Callable:
        """Creates closure that backpropagates either from output layer or from loss layer"""

        def closure() -> Tuple[Optional[torch.Tensor], torch.Tensor]:
            optimizer.zero_grad()
            output = model(data)
            if last_layer == "output":
                output.backward(torch.ones_like(target))
                return None, output
            elif last_layer == 'loss':
                loss = compute_loss(output - target)
                loss.backward()
                return loss, output
            else:
                assert False, 'last layer must be "output" or "loss"'

        return closure

    #    loss = compute_loss(output - Y.t())
    #    loss.backward()

    loss, output = optimizer.step(closure=backward('loss'))
    check_close(output.t(), [[-4, 2, 8]])
    check_close(residuals.t(), [[-4, 1, 6]])
    check_close(loss, 8.833333)

    # batch output Jacobian
    J = X.t()
    check_close(J, [[-2, -1], [0, 1], [2, 3]])

    # Hessian
    H = J.t() @ J / n
    check_close(H, [[2.66667, 2.66667], [2.66667, 3.66667]])

    # gradients, n,d
    G = residuals.repeat(1, d) * J
    check_close(G, [[8., 4.], [0., 1.], [12., 18.]])

    # mean gradient
    g = G.sum(dim=0) / n
    check_close(g, [6.66667, 7.66667])

    # empirical Fisher
    efisher = G.t() @ G / n
    check_close(efisher, [[69.3333, 82.6667], [82.6667, 113.667]])

    # centered empirical Fisher (Sigma in OpenAI paper, estimate of Sigma in Jain paper)
    sigma = efisher - outer(g, g)
    check_close(sigma, [[24.8889, 31.5556], [31.5556, 54.8889]])

    # loss
    loss2 = (residuals * residuals).sum() / (2 * n)
    check_close(toscalar(loss2), 8.83333)

    # TODO: get A's and B's, compute Sigma, Sigma_c, gradient diversity,
    sigma_norm = torch.norm(sigma)
    g_norm = torch.norm(g)

    g_ = g.unsqueeze(0)  # turn g into row matrix

    # predicted drop in loss if we take a Newton step
    excess = toscalar(g_ @ H.inverse() @ g_.t() / 2)
    check_close(excess, 8.83333)

    def loss_direction(direction, eps):
        """loss improvement if we take step eps in direction dir"""
        return toscalar(eps * (direction @ g.t()) - 0.5 * eps ** 2 * direction @ H @ direction.t())

    newtonImprovement = loss_direction(g_ @ H.inverse(), 1)
    check_close(newtonImprovement, 8.83333)

    ############################
    # OpenAI quantities
    grad_curvature = toscalar(g_ @ H @ g_.t())  # curvature in direction of g
    stepOpenAI = toscalar(g.norm() ** 2 / grad_curvature) if g_norm else 999
    check_close(stepOpenAI, 0.170157)
    batchOpenAI = toscalar(torch.trace(H @ sigma) / grad_curvature) if g_norm else 999
    check_close(batchOpenAI, 0.718603)

    # improvement in loss when we take gradient step with optimal learning rate
    gradientImprovement = loss_direction(g_, stepOpenAI)
    assert newtonImprovement > gradientImprovement
    check_close(gradientImprovement, 8.78199)

    ############################
    # Gradient diversity  quantities
    diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2
    check_close(diversity, 5.31862)

    ############################
    # Jain/Kakade quantities

    # noise scale (Jain, minimax rate of estimator)
    noise_variance = torch.trace(H.inverse() @ sigma)
    check_close(noise_variance, 26.)

    isqrtH = inv_square_root(to_numpy(H))
    isqrtH = torch.tensor(isqrtH)
    # measure of misspecification between model and actual noise (Jain, \rho)
    # formula (3) of "Parallelizing Stochastic Gradient Descent"
    p_sigma = (kron(H, torch.eye(d)) + kron(torch.eye(d), H)).inverse() @ vec(sigma)
    p_sigma = unvec(p_sigma, d)
    rho = d / rank(p_sigma) if sigma_norm > 0 else 1
    check_close(rho, 1.21987)

    rhoSimple = (d / rank(isqrtH @ sigma @ isqrtH)) if sigma_norm > 0 else 1
    check_close(rhoSimple, 1.4221)
    assert 1 <= rho <= d, rho

    # divergent learning rate for batch-size 1 (Jain)
    stepMin = 2 / torch.trace(H)
    check_close(stepMin, 0.315789)

    # divergent learning rate for batch-size infinity
    stepMax = 2 / l2_norm(H)
    check_close(stepMax, 0.340147)

    # divergent learning rate for batch-size 1, adjusted for misspecification
    check_close(stepMin / rhoSimple, 0.222058)
    check_close(stepMin / rho, 0.258871)

    # batch size that gives provides lr halfway between stepMin and stepMax
    batchJain = 1 + rank(H)
    check_close(batchJain, 2.07713)

    # batch size that provides halfway point after adjusting for misspecification
    check_close(1 + rank(H) * rhoSimple, 2.5318)
    check_close(1 + rank(H) * rho, 2.31397)

    loss, output = optimizer.step(closure=backward('output'))
    # TODO: get A's and B's, compute H, rho, Newton decrement, all learning rate + batch size stats
    #    print(loss.item())


def vec(a):
    """vec operator, stack columns of the matrix into single column matrix."""
    assert len(a.shape) == 2
    return a.t().reshape(-1, 1)


def unvec(a, rows):
    """reverse of vec, rows specifies number of rows in the final matrix."""
    assert len(a.shape) == 2
    assert a.shape[0] % rows == 0
    cols = a.shape[0] // rows
    return a.reshape(cols, -1).t()


def kron(a, b):
    return torch.einsum("ab,cd->acbd", a, b).view(a.size(0) * b.size(0), a.size(1) * b.size(1))


def l2_norm(mat):
    return max(torch.eig(mat).eigenvalues.flatten())


def inv_square_root(mat):
    assert type(mat) == np.ndarray
    return scipy.linalg.inv(scipy.linalg.sqrtm(mat))


def rank(mat):
    """Effective rank of matrix."""
    return torch.trace(mat) / l2_norm(mat)


def outer(x, y):
    return x.unsqueeze(1) @ y.unsqueeze(0)


def toscalar(x):
    if hasattr(x, 'item'):
        return x.item()
    x = to_numpy(x).flatten()
    assert len(x) == 1
    return x[0]


def to_numpy(x, dtype=np.float32):
    """Utility function to convert object to numpy array."""
    if hasattr(x, 'numpy'):  # PyTorch tensor
        return x.detach().numpy().astype(dtype)
    elif type(x) == np.ndarray:
        return x.astype(dtype)
    else:  # Some Python type
        return np.array(x).astype(dtype)


def check_close(observed, truth):
    truth = to_numpy(truth)
    observed = to_numpy(observed)
    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    np.testing.assert_allclose(truth, observed, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    test_loss()

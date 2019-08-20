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

    # matrix of activations, (n, d)
    A = model.w.data_input
    check_close(A, J)

    # matrix of backprops, add factor n to remove dependence on batch-size
    B = model.w.grad_output * n
    check_close(B, residuals)

    # gradients, n,d
    # method 1, manual computation
    G = residuals.repeat(1, d) * J
    check_close(G, [[8., 4.], [0., 1.], [12., 18.]])

    # method 2, get them of activation + backprop values
    check_close(G, khatri_rao_t(A, B))

    # method 3, PyTorch autograd
    # (n,) losses vector
    losses = torch.stack([compute_loss(r) for r in residuals])
    # batch-loss jacobian
    G2 = jacobian(losses, model.w.weight) * n
    # per-example gradients are row-matrices, squeeze to stack them into a single matrix
    G2 = G2.squeeze(1)
    check_close(G2, G)

    # Hessian

    # method 1, manual computation
    H = J.t() @ J / n
    check_close(H, [[2.66667, 2.66667], [2.66667, 3.66667]])

    # method 2, using activation + backprop values
    check_close(A.t() @ torch.eye(n) @ A / n, H)

    # method 3, PyTorch backprop
    hess = hessian(compute_loss(residuals), model.w.weight)
    hess = hess.squeeze(2)
    hess = hess.squeeze(0)
    check_close(hess, H)

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

    isqrtH = pinv_square_root(H)
    # measure of misspecification between model and actual noise (Jain, \rho)
    # formula (3) of "Parallelizing Stochastic Gradient Descent"
    p_sigma = (kron(H, torch.eye(d)) + kron(torch.eye(d), H)).inverse() @ vec(sigma)
    p_sigma = unvec(p_sigma, d)
    rho = d / rank(p_sigma) if sigma_norm > 0 else 1
    check_close(rho, 1.21987)

    rhoSimple = (d / rank(isqrtH @ sigma @ isqrtH)) if sigma_norm > 0 else 1
    check_close(rhoSimple, 1.4221)
    assert 1 <= rho <= d, rho

    # divergent learning rate for batch-size 1 (Jain). Approximates max||x_i|| with avg.
    # For more accurate results may want to add stddev of ||x_i||
    # noinspection PyTypeChecker
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

    # loss, output = optimizer.step(closure=backward('output'))
    # TODO: get A's and B's, compute H, rho, Newton decrement, all learning rate + batch size stats
    #    print(loss.item())


class Net2(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.W = nn.Linear(d1, d2, bias=False)
        self.X2t = nn.Linear(d2, 1, bias=False)

    def forward(self, X1: torch.Tensor):
        result = self.W(X1)
        result = self.X2t(result)
        return result


def test_multilayer():
    # Reproduce multilayer example
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/curvature-unit-tests.nb

    d1 = 2
    d2 = 4
    n = 3
    model = Net2(d1, d2)

    W0 = torch.tensor([[3, 3], [0, 3], [1, -1], [-3, 1]]).float()
    model.W.weight.data.copy_(W0)
    X2 = torch.tensor([[1], [-2], [-1], [3]]).float()
    assert X2.shape == (d2, 1)
    model.X2t.weight.data.copy_(X2.t())

    X1 = torch.tensor([[2, -2, 3], [-3, 1, -3]]).float()
    assert X1.shape == (d1, n)

    Y = torch.tensor([[-2, -3, 0]]).float()
    assert Y.shape == (1, n)

    data = X1.t()  # PyTorch expects batch dimension first
    target = Y.t()
    assert data.shape[0] == n

    output = model(data)
    # residuals, aka e
    residuals = output - Y.t()

    def compute_loss(residuals_):
        return torch.sum(residuals_ * residuals_) / (2 * n)

    loss = compute_loss(residuals)

    print(repr(loss.detach().numpy()))
    assert loss - 187.5 < 1e-5, torch.norm(loss) - 8.83333
    print(loss)

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0, momentum=0, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned")
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
    check_close(output.t(), [[-17, 15, -24]])
    check_close(residuals.t(), [[-15, 18, -24]])
    check_close(loss, 187.5)

    # batch output Jacobian, n rows, i'th row gives sensitivity of i'th output example to parameters
    J = kron(X1, X2).t()
    assert J.shape == (n, d1 * d2)
    check_close(J, [[2, -4, -2, 6, -3, 6, 3, -9], [-2, 4, 2, -6, 1, -2, -1, 3], [3, -6, -3, 9, -3, 6, 3, -9]])

    # matrix of activations, (n, d1)
    At = model.W.data_input
    A = At.t()
    check_close(At, X1.t())

    # matrix of backprops, add factor n to remove dependence on batch-size
    Bt = model.W.grad_output * n
    check_close(Bt, [[-15, 30, 15, -45], [18, -36, -18, 54], [-24, 48, 24, -72]])

    # gradients, n,d
    # method 1, manual computation
    G = khatri_rao_t(At, Bt)
    assert G.shape == (n, d1 * d2)
    check_close(G, [[-30, 60, 30, -90, 45, -90, -45, 135], [-36, 72, 36, -108, 18, -36, -18, 54],
                    [-72, 144, 72, -216, 72, -144, -72, 216]])

    # method 2, PyTorch autograd
    # (n,) losses vector
    losses = torch.stack([compute_loss(r) for r in residuals])
    # batch-loss jacobian
    G2 = jacobian(losses, model.W.weight) * n
    # per-example gradients are row-matrices, squeeze to stack them into a single matrix
    # each element of G2 is a matrix, vectorize+transpose to turn it into a row
    G2 = G2.transpose(1, 2).reshape(n, d1 * d2)
    check_close(G2, G)

    # Hessian

    # method 1, manual computation
    H = J.t() @ J / n
    check_close(H, [[5.66667, -11.3333, -5.66667, 17., -5.66667, 11.3333, 5.66667, -17.],
                    [-11.3333, 22.6667, 11.3333, -34., 11.3333, -22.6667, -11.3333, 34.],
                    [-5.66667, 11.3333, 5.66667, -17., 5.66667, -11.3333, -5.66667, 17.],
                    [17., -34., -17., 51., -17., 34., 17., -51.],
                    [-5.66667, 11.3333, 5.66667, -17., 6.33333, -12.6667, -6.33333, 19.],
                    [11.3333, -22.6667, -11.3333, 34., -12.6667, 25.3333, 12.6667, -38.],
                    [5.66667, -11.3333, -5.66667, 17., -6.33333, 12.6667, 6.33333, -19.],
                    [-17., 34., 17., -51., 19., -38., -19., 57.]])

    # method 2, using activation + upstream matrices
    check_close(kron(A @ A.t(), X2 @ X2.t()) / n, H)

    # method 3, PyTorch autograd
    hess = hessian(compute_loss(residuals), model.W.weight)
    hess = hess.squeeze(2)
    hess = hess.squeeze(0)
    hess = hess.transpose(2, 3).transpose(0, 1).reshape(d1 * d2, d1 * d2)
    check_close(hess, H)

    # method 4, get Jacobian + Hessian using backprop
    _loss, _output = optimizer.step(closure=backward('output'))
    B2t = model.W.grad_output

    # alternative way of getting batch Jacobian
    J2 = khatri_rao_t(At, B2t)
    check_close(J2, J)
    H2 = J2.t() @ J2 / n
    check_close(H2, H)

    # mean gradient
    g = G.sum(dim=0) / n
    check_close(g, [-46, 92, 46, -138, 45, -90, -45, 135])

    # empirical Fisher
    efisher = G.t() @ G / n
    check_close(efisher, [[2460, -4920, -2460, 7380, -2394, 4788, 2394, -7182],
                          [-4920, 9840, 4920, -14760, 4788, -9576, -4788, 14364],
                          [-2460, 4920, 2460, -7380, 2394, -4788, -2394, 7182],
                          [7380, -14760, -7380, 22140, -7182, 14364, 7182, -21546],
                          [-2394, 4788, 2394, -7182, 2511, -5022, -2511, 7533],
                          [4788, -9576, -4788, 14364, -5022, 10044, 5022, -15066],
                          [2394, -4788, -2394, 7182, -2511, 5022, 2511, -7533],
                          [-7182, 14364, 7182, -21546, 7533, -15066, -7533, 22599]])

    # centered empirical Fisher (Sigma in OpenAI paper, estimate of Sigma in Jain paper)
    sigma = efisher - outer(g, g)
    check_close(sigma, [[344, -688, -344, 1032, -324, 648, 324, -972], [-688, 1376, 688, -2064, 648, -1296, -648, 1944],
                        [-344, 688, 344, -1032, 324, -648, -324, 972],
                        [1032, -2064, -1032, 3096, -972, 1944, 972, -2916],
                        [-324, 648, 324, -972, 486, -972, -486, 1458], [648, -1296, -648, 1944, -972, 1944, 972, -2916],
                        [324, -648, -324, 972, -486, 972, 486, -1458],
                        [-972, 1944, 972, -2916, 1458, -2916, -1458, 4374]])

    # loss
    loss2 = (residuals * residuals).sum() / (2 * n)
    check_close(toscalar(loss2), 187.5)

    # TODO: get A's and B's, compute Sigma, Sigma_c, gradient diversity,
    sigma_norm = torch.norm(sigma)
    g_norm = torch.norm(g)

    g_ = g.unsqueeze(0)  # turn g into row matrix

    # predicted drop in loss if we take a Newton step
    excess = toscalar(g_ @ pinv(H) @ g_.t() / 2)
    check_close(excess, 187.456)

    def loss_direction(direction, eps):
        """loss improvement if we take step eps in direction dir"""
        return toscalar(eps * (direction @ g.t()) - 0.5 * eps ** 2 * direction @ H @ direction.t())

    newtonImprovement = loss_direction(g_ @ pinv(H), 1)
    check_close(newtonImprovement, 187.456)

    ############################
    # OpenAI quantities
    grad_curvature = toscalar(g_ @ H @ g_.t())  # curvature in direction of g
    stepOpenAI = toscalar(g.norm() ** 2 / grad_curvature) if g_norm else 999
    check_close(stepOpenAI, 0.00571855)
    batchOpenAI = toscalar(torch.trace(H @ sigma) / grad_curvature) if g_norm else 999
    check_close(batchOpenAI, 0.180201)

    # improvement in loss when we take gradient step with optimal learning rate
    gradientImprovement = loss_direction(g_, stepOpenAI)
    assert newtonImprovement > gradientImprovement
    check_close(gradientImprovement, 177.604)

    ############################
    # Gradient diversity  quantities
    diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2
    check_close(diversity, 3.6013)

    ############################
    # Jain/Kakade quantities

    # noise scale (Jain, minimax rate of estimator)
    noise_variance = torch.trace(pinv(H) @ sigma)
    check_close(noise_variance, 333.706)

    isqrtH = pinv_square_root(H)
    #    isqrtH = torch.tensor(isqrtH)
    # measure of misspecification between model and actual noise (Jain, \rho)
    # formula (3) of "Parallelizing Stochastic Gradient Descent"
    p_sigma = pinv(kron(H, torch.eye(d1*d2)) + kron(torch.eye(d1*d2), H)) @ vec(sigma)
    p_sigma = unvec(p_sigma, d1*d2)
    rho = d1*d2 / rank(p_sigma) if sigma_norm > 0 else 1
    check_close(rho, 6.48399)

    rhoSimple = (d1*d2 / rank(isqrtH @ sigma @ isqrtH)) if sigma_norm > 0 else 1
    check_close(rhoSimple, 6.55661)

    # divergent learning rate for batch-size 1 (Jain). Approximates max||x_i|| with avg.
    # For more accurate results may want to add stddev of ||x_i||
    # noinspection PyTypeChecker
    stepMin = 2 / torch.trace(H)
    check_close(stepMin, 0.0111111)

    # divergent learning rate for batch-size infinity
    stepMax = 2 / l2_norm(H)
    check_close(stepMax, 0.011419)

    # divergent learning rate for batch-size 1, adjusted for misspecification
    check_close(stepMin / rhoSimple, 0.00169464)
    check_close(stepMin / rho, 0.00171362)

    # batch size that gives provides lr halfway between stepMin and stepMax
    batchJain = 1 + rank(H)
    check_close(batchJain, 2.02771)

    # batch size that provides halfway point after adjusting for misspecification
    check_close(1 + rank(H) * rhoSimple, 7.73829)
    check_close(1 + rank(H) * rho, 7.66365)

    # loss, output = optimizer.step(closure=backward('output'))
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


def inv_square_root_numpy(mat):
    assert type(mat) == np.ndarray
    return scipy.linalg.inv(scipy.linalg.sqrtm(mat))


def pinv_square_root_numpy(mat):
    assert type(mat) == np.ndarray
    result = scipy.linalg.inv(scipy.linalg.sqrtm(mat))
    return result


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


def khatri_rao(A, B):
    """Khatri-Rao product, see
    Section 2.6 of Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM review 51.3 (2009): 455-500"""
    assert A.shape[1] == B.shape[1]
    return torch.einsum("ik,jk->ijk", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])


# Autograd functions, from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def test_khatri_rao():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[5, 12], [7, 16], [15, 24], [21, 32]])
    check_close(khatri_rao(A, B), C)


def khatri_rao_t(A, B):
    """Like Khatri-Rao, but iterators over rows of matrices instead of cols"""
    assert A.shape[0] == B.shape[0]
    return torch.einsum("ki,kj->kij", A, B).reshape(A.shape[0], A.shape[1] * B.shape[1])


def test_khatri_rao_t():
    A = torch.tensor([[-2., -1.],
                      [0., 1.],
                      [2., 3.]])
    B = torch.tensor([[-4.],
                      [1.],
                      [6.]])
    C = torch.tensor([[8., 4.],
                      [0., 1.],
                      [12., 18.]])
    check_close(khatri_rao_t(A, B), C)


def pinv(mat, eps=1e-4) -> torch.Tensor:
    """Computes pseudo-inverse of mat, treating eigenvalues below eps as 0."""

    # TODO(y): make eps scale invariant by diving by norm first
    u, s, v = torch.svd(mat)
    one = torch.from_numpy(np.array(1))
    ivals: torch.Tensor = one / s
    si = torch.where(s > eps, ivals, s)
    return u @ torch.diag(si) @ v.t()


def pinv_square_root(mat, eps=1e-4) -> torch.Tensor:
    u, s, v = torch.svd(mat)
    one = torch.from_numpy(np.array(1))
    ivals: torch.Tensor = one / torch.sqrt(s)
    si = torch.where(s > eps, ivals, s)
    return u @ torch.diag(si) @ v.t()


def check_close(observed, truth):
    truth = to_numpy(truth)
    observed = to_numpy(observed)
    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    np.testing.assert_allclose(truth, observed, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    test_multilayer()
    test_khatri_rao()
    test_khatri_rao_t()
    test_loss()

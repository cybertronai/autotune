import os
import sys

# import torch
import torch

import util as u

import torch.nn.functional as F


def test_to_logits():
    p = torch.tensor([0.2, 0.5, 0.3])
    u.check_close(p, F.softmax(u.to_logits(p), dim=0))
    u.check_close(p.unsqueeze(0), F.softmax(u.to_logits(p.unsqueeze(0)), dim=1))


def test_cross_entropy_soft():
    q = torch.tensor([0.4, 0.6]).unsqueeze(0).float()
    p = torch.tensor([0.7, 0.3]).unsqueeze(0).float()
    observed_logit = u.to_logits(p)

    # Compare against other loss functions
    # https://www.wolframcloud.com/obj/user-eac9ee2d-7714-42da-8f84-bec1603944d5/newton/logistic-hessian.nb

    loss1 = F.binary_cross_entropy(p[0], q[0])
    u.check_close(loss1, 0.865054)

    loss_fn = u.CrossEntropySoft()
    loss2 = loss_fn(observed_logit, q)
    u.check_close(loss2, loss1)

    loss3 = F.cross_entropy(observed_logit, torch.tensor([0]))
    u.check_close(loss3, loss_fn(observed_logit, torch.tensor([[1, 0.]])))

    # check gradient
    observed_logit.requires_grad = True
    grad = torch.autograd.grad(loss_fn(observed_logit, target=q), observed_logit)
    u.check_close(p - q, grad[0])

    # check Hessian
    observed_logit = u.to_logits(p)
    observed_logit.zero_()
    observed_logit.requires_grad = True
    hessian_autograd = u.hessian(loss_fn(observed_logit, target=q), observed_logit)
    hessian_autograd = hessian_autograd.reshape((p.numel(), p.numel()))
    p = F.softmax(observed_logit, dim=1)
    hessian_manual = torch.diag(p[0]) - p.t() @ p
    u.check_close(hessian_autograd, hessian_manual)


def test_symsqrt():
    mat = torch.reshape(torch.arange(9) + 1, (3, 3)).float() + torch.eye(3) * 5
    mat = mat + mat.t()  # make symmetric
    smat = u.symsqrt(mat)
    u.check_close(mat, smat @ smat.t())
    u.check_close(mat, smat @ smat)

    def randomly_rotate(X):
        """Randomly rotate d,n data matrix X"""
        d, n = X.shape
        z = torch.randn((d, d), dtype=X.dtype)
        q, r = torch.qr(z)
        d = torch.diag(r)
        ph = d / abs(d)
        rot_mat = q * ph
        return rot_mat @ X

    n = 20
    d = 10
    X = torch.randn((d, n))

    # embed in a larger space
    X = torch.cat([X, torch.zeros_like(X)])
    X = randomly_rotate(X)
    cov = X.t() @ X
    sqrt, rank = u.symsqrt(cov, return_rank=True)
    assert rank == d
    assert torch.allclose(sqrt @ sqrt, cov, atol=1e-5)


def kron_quadratic_form(H, dd):
    """dd @ H @ dd.t(),"""
    pass


def kron_trace(H):
   """trace(H)"""
   pass

def kron_matmul(H, sigma):
    # H @ sigma
    pass


def kron_trace_matmul(H, sigma):
    # tr(H @ sigma)
    pass

def kron_pinv(H):
    # u.kron_pinv(H)
    pass


def kron_nan_check(H):
    pass
    # u.kron_nan_check(H)

def kron_fro_norm(H):
    pass
    # s.invH_fro = u.kron_fro_norm(invH)

def kron_sym_l2_norm(H):
    pass
    #                 s.H_l2 = u.kron_sym_l2_norm(H)

def kron_inv():
    pass
   # .kron_inverse(H)

def kron_sigma(G):
    # compute covariance # kron_sigma(G): computes covariance of gradients in kron form # G.t() @ G / n
    pass

def kron_batch_sum(G):
    # sums up over batch dimension to get average gradient
    pass


if __name__ == '__main__':
    u.run_all_tests(sys.modules[__name__])

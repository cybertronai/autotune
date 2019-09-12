import os
import sys

# import torch
import torch

import numpy as np
import util as u

import torch.nn.functional as F


def test_khatri_rao():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[5, 12], [7, 16],
                      [15, 24], [21, 32]])
    u.check_equal(u.khatri_rao(A, B), C)


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
    u.check_equal(u.khatri_rao_t(A, B), C)


def test_to_logits():
    torch.set_default_dtype(torch.float32)

    p = torch.tensor([0.2, 0.5, 0.3])
    u.check_close(p, F.softmax(u.to_logits(p), dim=0))
    u.check_close(p.unsqueeze(0), F.softmax(u.to_logits(p.unsqueeze(0)), dim=1))


def test_cross_entropy_soft():
    torch.set_default_dtype(torch.float32)

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
    u.seed_random(1)
    torch.set_default_dtype(torch.float32)

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
    cov = X @ X.t()
    sqrt, rank = u.symsqrt(cov, return_rank=True)
    assert rank == d
    assert torch.allclose(sqrt @ sqrt, cov, atol=1e-5)


def test_symsqrt_neg():
    """Test robustness to small negative eigenvalues."""
    u.seed_random(1)
    torch.set_default_dtype(torch.float32)

    mat = torch.tensor([[1.704692840576171875e-05, -9.693153669044357601e-15, -4.637238930627063382e-07,
                         -5.784777457051859528e-08, -7.958237541183521557e-11, -9.898678399622440338e-06,
                         -2.152719247305867611e-07, -1.635662982835128787e-08, -6.400216989277396351e-06,
                         -1.906904145698717912e-08],
                        [-9.693153669044357601e-15, 9.693318840469072190e-15, -4.495100185314538545e-21,
                         -5.607465147510056466e-22, -7.714305304820877864e-25, -9.595268609986125189e-20,
                         -2.086734953246380139e-21, -1.585527368218132061e-22, -6.204040063550678436e-20,
                         -1.848454455492805625e-22],
                        [-4.637238930627063382e-07, -4.495100185314538545e-21, 4.637315669242525473e-07,
                         -2.682631101578927119e-14, -3.690551006200058401e-17, -4.590410516980281130e-12,
                         -9.983013764605641605e-14, -7.585219006177833234e-15, -2.968034672201635971e-12,
                         -8.843071403179941781e-15],
                        [-5.784777457051859528e-08, -5.607465147510056466e-22, -2.682631101578927119e-14,
                         5.784875867220762302e-08, -4.603820523722603687e-18, -5.726360683376563454e-13,
                         -1.245342652107404510e-14, -9.462269640040017228e-16, -3.702509490405986314e-13,
                         -1.103139288087268827e-15],
                        [-7.958237541183521557e-11, -7.714305304820877864e-25, -3.690551006200058401e-17,
                         -4.603820523722603687e-18, 7.958373543504038139e-11, -7.877872806981575052e-16,
                         -1.713243580702275093e-17, -1.301743849346897150e-18, -5.093618864028342207e-16,
                         -1.517611416046059107e-18],
                        [-9.898678399622440338e-06, -9.595268609986125189e-20, -4.590410516980281130e-12,
                         -5.726360683376563454e-13, -7.877872806981575052e-16, 9.898749340209178627e-06,
                         -2.130980288062023220e-12, -1.619145491510778911e-13, -6.335585528427500890e-11,
                         -1.887647481900456281e-13],
                        [-2.152719247305867611e-07, -2.086734953246380139e-21, -9.983013764605641605e-14,
                         -1.245342652107404510e-14, -1.713243580702275093e-17, -2.130980288062023220e-12,
                         2.152755484985391377e-07, -3.521243228436451468e-15, -1.377834044774539635e-12,
                         -4.105169319306963341e-15],
                        [-1.635662982835128787e-08, -1.585527368218132061e-22, -7.585219006177833234e-15,
                         -9.462269640040017228e-16, -1.301743849346897150e-18, -1.619145491510778911e-13,
                         -3.521243228436451468e-15, 1.635690871637507371e-08, -1.046895521119271810e-13,
                         -3.119158858896762436e-16],
                        [-6.400216989277396351e-06, -6.204040063550678436e-20, -2.968034672201635971e-12,
                         -3.702509490405986314e-13, -5.093618864028342207e-16, -6.335585528427500890e-11,
                         -1.377834044774539635e-12, -1.046895521119271810e-13, 6.400285201380029321e-06,
                         -1.220501699922618699e-13],
                        [-1.906904145698717912e-08, -1.848454455492805625e-22, -8.843071403179941781e-15,
                         -1.103139288087268827e-15, -1.517611416046059107e-18, -1.887647481900456281e-13,
                         -4.105169319306963341e-15, -3.119158858896762436e-16, -1.220501699922618699e-13,
                         1.906936653028878936e-08]])
    evals = torch.eig(mat).eigenvalues
    assert torch.min(evals) < 0
    smat = u.symsqrt(mat)
    u.check_close(mat, smat @ smat.t())
    u.check_close(mat, smat @ smat)


def test_truncated_lyapunov():
    d = 100
    n = 1000
    shared_rank = 2
    independent_rank = 1
    A, C = u.random_cov_pair(shared_rank=shared_rank, independent_rank=independent_rank, strength=0.1, d=d, n=n)
    X = u.truncated_lyapunov(A, C)

    # effective rank of X captures dimensionality of shared subspace
    u.check_close(u.rank(X), shared_rank+independent_rank, rtol=1e-4)
    u.check_close(u.erank(X), shared_rank, rtol=1e-2)


def test_robust_svd():
    mat = np.genfromtxt('test/gesvd_crash.txt', delimiter=",").astype(np.float32)
    mat = torch.tensor(mat)
    U, S, V = u.robust_svd(mat)
    mat2 = U @ torch.diag(S) @ V.T
    print(torch.norm(mat-mat2))


if __name__ == '__main__':
    test_truncated_lyapunov()
    # test_robust_svd()
    #u.run_all_tests(sys.modules[__name__])

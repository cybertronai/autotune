import os
import sys

# import torch
import torch

sys.path.insert(0, os.environ['HOME'] + '/git0/pytorch-curv/examples/cifar')
import util as u

import torch.nn.functional as F


def to_logits_test():
    p = torch.tensor([0.2, 0.5, 0.3])
    u.check_close(p, F.softmax(u.to_logits(p), dim=0))
    u.check_close(p.unsqueeze(0), F.softmax(u.to_logits(p.unsqueeze(0)), dim=1))


def cross_entropy_soft_test():
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


if __name__ == '__main__':
    u.run_all_tests(sys.modules[__name__])

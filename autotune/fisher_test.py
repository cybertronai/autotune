import sys
from typing import Callable

import torch
import torch.nn as nn

from collections import defaultdict

import torch.nn.functional as F

import autograd_hacks
import autograd_lib

import util as u


def simple_model(d, num_layers, identity_init=False):
    """Creates simple linear neural network initialized to identity"""
    layers = []
    for i in range(num_layers):
        layer = nn.Linear(d, d, bias=True)
        if identity_init:
            layer.weight.data.copy_(torch.eye(d))
        layers.append(layer)
    obj = torch.nn.Sequential(*layers)
    setattr(obj, "layers", layers)
    return obj


class TinyNet(nn.Module):
    def __init__(self, nonlin=True):
        super(TinyNet, self).__init__()
        self.nonlin = nonlin
        self.conv1 = nn.Conv2d(1, 2, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 2, 1)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 10)
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):            # 28x28
        x = F.max_pool2d(x, 4, 4)    # 7x7
        x = self.conv1(x)    # 6x6
        if self.nonlin:
            x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)    # 3x3
        x = self.conv2(x)    # 2x2
        if self.nonlin:
            x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)    # 1x1
        x = x.view(-1, 2 * 1 * 1)    # C * W * H
        x = self.fc1(x)
        if self.nonlin:
            x = F.relu(x)
        x = self.fc2(x)
        return x


def test_diag_conv():
    """Test computing quantities of diagonal of empirical Fisher.

    Einsum optimizations: https://colab.research.google.com/drive/16nKr_LmiiH8pgGkF1gNNahK83WVCpqk4#scrollTo=GxySBytnyx4a
    """

    u.seed_random(1)
    supported_layers = ('Linear', 'Conv2d')

    model = TinyNet(nonlin=False)
    n = 3
    autograd_lib.register(model)

    data = torch.rand(n, 1, 28, 28)
    targets = torch.LongTensor(n).random_(0, 10)

    # old way of computing per-example gradients
    autograd_hacks.add_hooks(model)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    output = model(data)
    loss = loss_fn(output, targets)
    loss.backward()
    autograd_hacks.compute_grad1(model, loss_type='sum')

    cov_diag_old = defaultdict(float)
    norms_old = defaultdict(float)
    trace_old = defaultdict(float)
    cov_diag_bias_old = defaultdict(float)
    norms_bias_old = defaultdict(float)
    trace_bias_old = defaultdict(float)
    for layer in model.layers:
        if autograd_lib._layer_type(layer) not in supported_layers:
            continue
        assert (torch.allclose(layer.weight.grad1.sum(dim=0), layer.weight.grad))
        cov_diag_old[layer] = u.square(layer.weight.grad1).sum(dim=0)
        dims = list(range(len(layer.weight.grad1.shape)))
        norms_old[layer] = u.square(layer.weight.grad1).sum(dim=dims[1:])
        trace_old[layer] = u.square(layer.weight.grad1).sum()
        cov_diag_bias_old[layer] = u.square(layer.bias.grad1).sum(dim=0)

        dims = list(range(len(layer.bias.grad1.shape)))
        norms_bias_old[layer] = u.square(layer.bias.grad1).sum(dim=dims[1:])
        trace_bias_old[layer] = u.square(layer.bias.grad1).sum()

    # new way of computing per-example gradients
    activations = {}

    def save_activations(layer, A, _):
        activations[layer] = A.detach()

    with autograd_lib.module_hook(save_activations):
        loss = loss_fn(model(data), targets)

    cov_diag_new = defaultdict(float)
    norms_new = defaultdict(float)
    trace_new = defaultdict(float)
    cov_diag_bias_new = defaultdict(float)
    norms_bias_new = defaultdict(float)
    trace_bias_new = defaultdict(float)

    ein = torch.einsum
    def compute_diag(layer, _, B):
        layer_type = autograd_lib._layer_type(layer)
        if layer_type not in supported_layers:
            return
        A = activations[layer]
        n = A.shape[0]

        if layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels
            Oh, Ow = B.shape[-2:]
            Ih, Iw = A.shape[-2:]
            assert tuple(B.shape) == (n, do, Oh, Ow)
            assert tuple(A.shape) == (n, di, Ih, Iw)

            A = F.unfold(A, (Kh, Kw))                      # n, di * Kh * Kw, Oh * Ow
            assert tuple(A.shape) == (n, di * Kh * Kw, Oh * Ow)
            B = B.reshape(n, do, -1)                       # n, do, Oh * Ow

            result_slow = torch.einsum('nlp,nip,nlq,niq->li', B, A, B, A)  # do, di * Kh * Kw
            result_fast = u.square(torch.einsum('nlp,nip->nli', B, A)).sum(dim=0)  # O(n,d_in,d_out)
            u.check_close(result_slow, result_fast)
            cov_diag_new[layer] += result_fast.reshape(layer.weight.shape)      # do, di, Kh, Kw

            result_slow = torch.einsum('nlp,nlq->l', B, B)  # do,
            result_fast = u.square(ein('nlp->nl', B)).sum(dim=0)    # manual optimization https://github.com/dgasmith/opt_einsum/issues/112
            u.check_close(result_slow, result_fast)
            cov_diag_bias_new[layer] += result_fast.reshape(layer.bias.shape)     # do, di, Kh, Kw

            # norms
            # result_slow = torch.einsum('nlp,nip,nlq,niq->n', B, A, B, A)  # do, di * Kh * Kw
            # result_fast = ein('nqp,nqp->n', ein('nlq,nlp->nqp', B, B), ein('niq,nip->nqp', A, A))  # O(n * patch^2)
            result_faster = ein('nli->n', u.square(ein('nlq,niq->nli', B, A)))                     # O(n * d_in * d_out)
            norms_new[layer] += result_faster

            # result_slow = torch.einsum("nlp,nlq->n", B, B)
            result_fast = u.square(ein('nlp->nl', B)).sum(dim=1)
            norms_bias_new[layer] += result_fast

            # trace
            # result_slow = torch.einsum('nlp,nip,nlq,niq->', B, A, B, A)  # do, di * Kh * Kw
            result_fast = u.square(ein('nlq,niq->nli', B, A)).sum()
            trace_new[layer] += result_fast

            # result_slow = torch.einsum('nlp,nlq->', B, B)  # do, di * Kh * Kw
            result_fast = u.square(ein('nlp->nl', B)).sum()
            trace_bias_new[layer] += result_fast

        elif layer_type == 'Linear':
            cov_diag_new[layer] += torch.einsum("ni,nk->ki", A * A, B * B)
            u.check_close(cov_diag_new[layer], torch.einsum("ni,ni,nk,nk->ki", A, A, B, B))

            norms_new[layer] += (B * B).sum(dim=1) * (A * A).sum(dim=1)
            u.check_close(norms_new[layer], torch.einsum("ni,ni,nk,nk->n", A, A, B, B))

            trace_new[layer] += ((B * B).sum(dim=1) * (A * A).sum(dim=1)).sum(dim=0)
            u.check_close(trace_new[layer], torch.einsum("ni,ni,nk,nk->", A, A, B, B))

            cov_diag_bias_new[layer] += torch.einsum("nk->k", B * B)
            u.check_close(cov_diag_bias_new[layer], torch.einsum("nk,nk->k", B, B))

            norms_bias_new[layer] += (B * B).sum(dim=1)
            u.check_close(norms_bias_new[layer], torch.einsum("nk,nk->n", B, B))

            trace_bias_new[layer] += (B * B).sum()
            u.check_close(trace_bias_new[layer], torch.einsum("nk,nk->", B, B))

    with autograd_lib.module_hook(compute_diag):
        loss.backward()

    for layer in model.layers:
        if autograd_lib._layer_type(layer) not in supported_layers:
            continue
        u.check_close(cov_diag_new[layer], cov_diag_old[layer])
        u.check_close(cov_diag_bias_new[layer], cov_diag_bias_old[layer])
        u.check_close(norms_new[layer], norms_old[layer])
        u.check_close(norms_bias_new[layer], norms_bias_old[layer])
        u.check_close(trace_new[layer], trace_old[layer])
        u.check_close(trace_bias_new[layer], trace_bias_old[layer])


if __name__ == '__main__':
    test_diag()
    test_diag_conv()

# Tests that compare manual computation of quantities against PyTorch autograd

import os
import sys

import globals as gl
import torch
import wandb
from attrdict import AttrDefault
from torch import optim
from torch.utils.tensorboard import SummaryWriter

module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path)
import util as u

import numpy as np

unfold = torch.nn.functional.unfold
fold = torch.nn.functional.fold


def autoencoder_minimize_test():
    """Minimize autoencoder for a few steps."""
    data_width = 4
    targets_width = 2

    batch_size = 64
    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    model = u.SimpleFullyConnected([d1, d2, d3], nonlin=True)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    loss = 0
    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        if i == 0:
            assert loss > 0.055
            pass
        loss.backward()
        optimizer.step()

    assert loss < 0.032


def autoencoder_newton_test():
    """Use Newton's method to train autoencoder."""

    image_size = 3
    batch_size = 64
    dataset = u.TinyMNIST('/tmp', download=True, data_width=image_size, targets_width=image_size,
                          dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d = image_size ** 2  # hidden layer size
    u.seed_random(1)
    model = u.SimpleFullyConnected([d, d])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        if i > 0:
            assert loss < 1e-9

        loss.backward()
        W = model.layers[0].weight
        grad = u.tvec(W.grad)

        loss = loss_fn(model(data), targets)
        H = u.hessian(loss, W)

        #  for col-major: H = H.transpose(0, 1).transpose(2, 3).reshape(d**2, d**2)
        H = H.reshape(d ** 2, d ** 2)

        #  For col-major: W1 = u.unvec(u.vec(W) - u.pinv(H) @ grad, d)
        W1 = u.untvec(u.tvec(W) - grad @ u.pinv(H), d)
        W.data.copy_(W1)


# main test example to fork for checking Hessians against autograd
def main_autograd_test():
    log_wandb = False
    autograd_check = True

    logdir = u.get_unique_logdir('/tmp/autoencoder_test/run')

    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)

    batch_size = 5
    u.seed_random(1)

    try:
        if log_wandb:
            wandb.init(project='test-autograd_test', name=run_name)
            wandb.tensorboard.patch(tensorboardX=False)
            wandb.config['batch'] = batch_size
    except Exception as e:
        print(f"wandb crash with {e}")

    data_width = 4
    targets_width = 2

    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    n = batch_size
    d = [d1, d2, d3]
    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=True)
    train_steps = 3

    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size * train_steps)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    for layer in model.layers:
        layer.register_forward_hook(u.capture_activations)
        layer.register_backward_hook(u.capture_backprops)
        layer.weight.register_hook(u.save_grad(layer.weight))

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    loss_hessian = u.HessianExactSqrLoss()

    gl.token_count = 0
    for train_step in range(train_steps):
        data, targets = next(train_iter)

        # get gradient values
        model.skip_backward_hooks = False
        model.skip_forward_hooks = False
        u.clear_backprops(model)
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward(retain_graph=True)
        model.skip_forward_hooks = True

        output = model(data)
        for bval in loss_hessian(output):
            output.backward(bval, retain_graph=True)

        model.skip_backward_hooks = True

        for (i, layer) in enumerate(model.layers):

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops_list[0] * n
            assert B_t.shape == (n, d[i + 1])

            G = u.khatri_rao_t(B_t, A_t)
            assert G.shape == (n, d[i] * d[i + 1])

            # average gradient
            g = G.sum(dim=0, keepdim=True) / n
            assert g.shape == (1, d[i] * d[i + 1])

            if autograd_check:
                u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
                u.check_close(g.reshape(d[i + 1], d[i]), layer.weight.saved_grad)

            # empirical Fisher
            efisher = G.t() @ G / n
            _sigma = efisher - g.t() @ g

            #############################
            # Hessian stats
            #############################
            A_t = layer.activations
            Bh_t = [layer.backprops_list[out_idx + 1] for out_idx in range(o)]
            Amat_t = torch.cat([A_t] * o, dim=0)  # todo: can instead replace with a khatri-rao loop
            Bmat_t = torch.cat(Bh_t, dim=0)

            assert Amat_t.shape == (n * o, d[i])
            assert Bmat_t.shape == (n * o, d[i + 1])

            # hessian in in row-vectorized layout instead of usual column vectorized, for easy comparison with PyTorch autograd
            Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch Jacobian
            H = Jb.t() @ Jb / n

            if autograd_check:
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                H_autograd = u.hessian(loss, layer.weight)
                u.check_close(H, H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1]))


# main test example to fork for checking Hessians against autograd
def subsampled_hessian_test():
    autograd_check = True

    batch_size = 5
    u.seed_random(1)

    data_width = 4
    targets_width = 2

    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    n = batch_size
    d = [d1, d2, d3]
    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=True)
    train_steps = 3

    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size * train_steps)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    for layer in model.layers:
        layer.register_forward_hook(u.capture_activations)
        layer.register_backward_hook(u.capture_backprops)
        layer.weight.register_hook(u.save_grad(layer.weight))

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    loss_hessian_approx = u.HessianSampledSqrLoss(samples=1)
    loss_hessian = u.HessianExactSqrLoss()

    data, targets = next(train_iter)

    # get gradient values
    model.skip_backward_hooks = False
    model.skip_forward_hooks = False
    u.clear_backprops(model)
    output = model(data)
    loss = loss_fn(output, targets)
    loss.backward(retain_graph=True)
    model.skip_forward_hooks = True

    # get exact Hessian
    output = model(data)
    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)
    model.skip_backward_hooks = True
    i, layer = next(enumerate(model.layers))

    #############################
    # Gradient stats
    #############################
    A_t = layer.activations
    assert A_t.shape == (n, d[i])

    # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
    B_t = layer.backprops_list[0] * n
    assert B_t.shape == (n, d[i + 1])

    G = u.khatri_rao_t(B_t, A_t)
    assert G.shape == (n, d[i] * d[i + 1])

    # average gradient
    g = G.sum(dim=0, keepdim=True) / n
    assert g.shape == (1, d[i] * d[i + 1])

    u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
    u.check_close(g.reshape(d[i + 1], d[i]), layer.weight.saved_grad)

    A_t = layer.activations
    Bh_t = [layer.backprops_list[out_idx + 1] for out_idx in range(o)]
    Amat_t = torch.cat([A_t] * o, dim=0)  # todo: can instead replace with a khatri-rao loop
    Bmat_t = torch.cat(Bh_t, dim=0)

    assert Amat_t.shape == (n * o, d[i])
    assert Bmat_t.shape == (n * o, d[i + 1])

    # hessian in in row-vectorized layout instead of usual column vectorized, for easy comparison with PyTorch autograd
    Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch Jacobian
    H = Jb.t() @ Jb / n

    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, targets)
    H_autograd = u.hessian(loss, layer.weight)
    u.check_close(H, H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1]))


def unfold_test():
    """ Test convolution as a special case of matrix multiplication with unfolded input tensors
    """
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0

    N, Xc, Xh, Xw = 1, 2, 3, 3
    model = u.SimpleConvolutional([Xc, 2])

    weight_buffer = model.layers[0].weight.data
    weight_buffer.copy_(torch.ones_like(weight_buffer))
    dims = N, Xc, Xh, Xw

    size = np.prod(dims)
    X = torch.range(0, size - 1).reshape(*dims)

    def loss_fn(data):
        err = data.reshape(len(data), -1)
        return torch.sum(err * err) / 2 / len(data)

    layer = model.layers[0]
    layer.register_forward_hook(u.capture_activations)
    layer.register_backward_hook(u.capture_backprops)
    output = model(X)
    loss = loss_fn(output)
    loss.backward()

    u.check_close(layer.activations, X)

    assert layer.backprops_list[0].shape == layer.output.shape

    unfold = torch.nn.functional.unfold
    fold = torch.nn.functional.fold
    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (2, 2))
    u.check_close(fold(out_unf, layer.output.shape[2:], (1, 1)), output)

    print("activations check passed")


# noinspection PyUnresolvedReferences
def conv_grad_test():
    """Test gradient computation for convolutional layer."""
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0
    N, Xc, Xh, Xw = 1, 2, 3, 3
    dd = [Xc, 2]
    model: u.SimpleConvolutional = u.SimpleConvolutional(dd)
    Kh, Kw = 2, 2
    Oh, Ow = Xh - Kh + 1, Xw - Kw + 1

    weight_buffer = model.layers[0].weight.data

    assert weight_buffer.shape == (dd[1], dd[0], Kh, Kw)

    # first output channel=1's, second channel=2's
    weight_buffer[0, :, :, :].copy_(torch.ones_like(weight_buffer[0, :, :, :]))
    weight_buffer[1, :, :, :].copy_(2 * torch.ones_like(weight_buffer[1, :, :, :]))

    dims = N, Xc, Xh, Xw

    size = np.prod(dims)
    X = torch.range(0, size - 1).reshape(*dims)

    def loss_fn(data):
        print(len(data))
        err = data.reshape(len(data), -1)
        return torch.sum(err * err) / 2 / len(data)

    layer = model.layers[0]
    layer.register_forward_hook(u.capture_activations)
    layer.register_backward_hook(u.capture_backprops)
    output = model(X)
    loss = loss_fn(output)
    loss.backward()

    u.check_equal(layer.activations, X)

    assert layer.backprops_list[0].shape == layer.output.shape

    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (2, 2))
    u.check_close(fold(out_unf, layer.output.shape[2:], (1, 1)), output)

    assert unfold(layer.activations, (Oh, Ow)).shape == (N, Xc * Kh * Kw, Oh * Ow)
    assert layer.backprops_list[0].shape == (N, dd[1], Oh, Ow)

    # make patches be the inner dimension
    bp = layer.backprops_list[0]
    bp = bp.reshape(N, dd[1], Oh * Ow)
    bp = bp.transpose(1, 2)

    print('backprops')
    print(bp)
    grad_unf = unfold(layer.activations, (Oh, Ow)) @ bp
    assert grad_unf.shape == (N, dd[0] * Kh * Kw, dd[1])  # need (dd[1], dd[0], Kh, Kw)
    grad_unf = grad_unf.transpose(1, 2)
    grads = grad_unf.reshape((N, dd[1], dd[0], Kh, Kw))
    assert N == 1, "currently only works for N=1"
    print('predicted')
    print(grads[0])
    print('actual')
    print(layer.weight.grad)
    print(torch.max(grads[0] - layer.weight.grad))
    u.check_equal(grads[0], layer.weight.grad)
    print("grad check passed")


def conv_multiexample_test():
    """Test per-example gradient computation for conv layer."""
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0
    u.seed_random(1)
    N, Xc, Xh, Xw = 3, 2, 3, 7
    dd = [Xc, 2]

    Kh, Kw = 2, 3
    Oh, Ow = Xh - Kh + 1, Xw - Kw + 1
    model = u.SimpleConvolutional(dd, kernel_size=(Kh, Kw)).double()

    weight_buffer = model.layers[0].weight.data

    # output channels, input channels, height, width
    assert weight_buffer.shape == (dd[1], dd[0], Kh, Kw)

    input_dims = N, Xc, Xh, Xw
    size = np.prod(input_dims)
    X = torch.arange(0, size).reshape(*input_dims).double()

    def loss_fn(data):
        err = data.reshape(len(data), -1)
        return torch.sum(err * err) / 2 / len(data)

    layer = model.layers[0]
    layer.register_forward_hook(u.capture_activations)
    layer.register_backward_hook(u.capture_backprops)
    output = model(X)
    loss = loss_fn(output)
    loss.backward()

    u.check_equal(layer.activations, X)

    assert layer.backprops_list[0].shape == layer.output.shape
    assert layer.output.shape == (N, dd[1], Oh, Ow)

    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (Kh, Kw))
    u.check_equal(fold(out_unf, (Oh, Ow), (1, 1)), output)
    u.check_equal(out_unf.view(N, dd[1], Oh, Ow), output)

    #    print(unfold(layer.activations, (Kh, Kw)))
    assert unfold(layer.activations, (Kh, Kw)).shape == (N, Xc * Kh * Kw, Oh * Ow)
    assert layer.backprops_list[0].shape == (N, dd[1], Oh, Ow)

    bp = layer.backprops_list[0] * N  # remove factor of N applied during loss batch averaging

    # merge patches into single dimension, move patches to be the inner dimension, output channels the rightmost dimension
    # since we multiplying on the left. TODO(y) try einsum for this and benchmark
    bp = bp.reshape(N, dd[1], Oh * Ow)
    bp = bp.transpose(1, 2)

    grad_unf = unfold(layer.activations, (Kh, Kw)) @ bp
    assert grad_unf.shape == (N, dd[0] * Kh * Kw, dd[1])

    # For comparison with autograd, shape needs to be (N, dd[1], dd[0], Kh, Kw)
    # therefore move output channels to the left, and unmerge remaining shapes
    grad_unf = grad_unf.transpose(1, 2)
    grads = grad_unf.reshape((N, dd[1], dd[0], Kh, Kw))
    mean_grad = torch.sum(grads, dim=0) / N

    print(f'grad: {torch.max(abs(mean_grad - layer.weight.grad))}')
    u.check_equal(mean_grad, layer.weight.grad)

    # compute per-example gradients using autograd, compare against manual computation
    for i in range(N):
        u.clear_backprops(model)
        output = model(X[i:i + 1, ...])
        loss = loss_fn(output)
        loss.backward()
        print(f'grad {i}: {torch.max(abs(grads[i] - layer.weight.grad))}')
        u.check_equal(grads[i], layer.weight.grad)


if __name__ == '__main__':
    subsampled_hessian_test()
    sys.exit()
    u.run_all_tests(sys.modules[__name__])

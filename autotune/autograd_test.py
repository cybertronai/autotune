# Tests that compare manual computation of quantities against PyTorch autograd

import os
import sys

import globals as gl
import torch
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import util as u

import numpy as np

unfold = torch.nn.functional.unfold
fold = torch.nn.functional.fold


def test_autoencoder_minimize():
    """Minimize autoencoder for a few steps."""
    u.seed_random(1)
    data_width = 4
    targets_width = 2

    batch_size = 64
    dataset = u.TinyMNIST(data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    model = u.SimpleFullyConnected([d1, d2, d3], nonlin=True)
    model.disable_hooks()

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
            assert loss > 0.054
            pass
        loss.backward()
        optimizer.step()

    assert loss < 0.0398


def test_autoencoder_newton():
    """Use Newton's method to train autoencoder."""

    image_size = 3
    batch_size = 64
    dataset = u.TinyMNIST(data_width=image_size, targets_width=image_size,
                          dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d = image_size ** 2  # hidden layer size
    u.seed_random(1)
    model = u.SimpleFullyConnected([d, d])
    model.disable_hooks()
    
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


def test_main_autograd():
    u.seed_random(1)
    log_wandb = False
    autograd_check = True
    use_double = False

    logdir = u.get_unique_logdir('/tmp/autoencoder_test/run')

    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)

    batch_size = 5

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
    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=True, bias=True)
    if use_double:
        model = model.double()
    train_steps = 3

    dataset = u.TinyMNIST(data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size * train_steps)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    loss_hessian = u.HessianExactSqrLoss()

    gl.token_count = 0
    for train_step in range(train_steps):
        data, targets = next(train_iter)
        if use_double:
            data, targets = data.double(), targets.double()

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
            if use_double:
                bval = bval.double()
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

            # per example gradients
            G = u.khatri_rao_t(B_t, A_t)
            assert G.shape == (n, d[i] * d[i + 1])
            Gbias = B_t
            assert Gbias.shape == (n, d[i + 1])
            
            # average gradient
            g = G.sum(dim=0, keepdim=True) / n
            gb = Gbias.sum(dim=0, keepdim=True) / n
            assert g.shape == (1, d[i] * d[i + 1])
            assert gb.shape == (1, d[i + 1])

            if autograd_check:
                u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
                u.check_close(g.reshape(d[i + 1], d[i]), layer.weight.saved_grad)
                u.check_close(torch.mean(B_t, dim=0), layer.bias.saved_grad)
                
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
            Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch output Jacobian
            H = Jb.t() @ Jb / n

            Hbias = Bmat_t.t() @ Bmat_t / n

            if autograd_check:
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                H_autograd = u.hessian(loss, layer.weight)
                Hbias_autograd = u.hessian(loss, layer.bias)
                u.check_close(H, H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1]))
                u.check_close(Hbias, Hbias_autograd)


def test_hessian():
    """Tests of Hessian computation."""
    u.seed_random(1)
    batch_size = 500

    data_width = 4
    targets_width = 4

    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    N = batch_size
    d = [d1, d2, d3]

    dataset = u.TinyMNIST(data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)
    data, targets = next(train_iter)

    def loss_fn(data, targets):
        assert len(data) == len(targets)
        err = data - targets.view(-1, data.shape[1])
        return torch.sum(err * err) / 2 / len(data)
    
    u.seed_random(1)
    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=False)
    
    # backprop hessian and compare against autograd
    loss_hessian = u.HessianExactSqrLoss()
    output = model(data)
    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)

    i, layer = next(enumerate(model.layers))
    A_t = layer.activations
    Bh_t = layer.backprops_list
    H = u.hessian_from_backprops(A_t, Bh_t)

    model.disable_hooks()
    H_autograd = u.hessian(loss_fn(model(data), targets), layer.weight)
    u.check_close(H, H_autograd.reshape(d[i+1] * d[i], d[i+1] * d[i]),
                  rtol=1e-4, atol=1e-7)

    # check first few per-example Hessians
    # todo: rename "samples" to num_samples
    assert len(Bh_t) == loss_hessian.samples
    Hs = u.per_example_hess(A_t, Bh_t)
    u.check_close(H, Hs.mean(dim=0))
    
    for xi in range(5):
        print("Checking Hessian", xi)
        loss = loss_fn(model(data[xi:xi + 1, ...]), targets[xi:xi+1])
        H_autograd = u.hessian(loss, layer.weight)
        u.check_close(Hs[xi], H_autograd.reshape(d[i+1]*d[i], d[i+1]*d[i]))

        # TODO: add bias calculation
        # Hbias_autograd = u.hessian(loss, layer.bias)
        # u.check_(Hs_bias[i], Hbias_autograd)
        # TODO: add bias
        #        loss.backward()
        #        u.check_equal(grads[i], layer.weight.grad)
        #        u.check_equal(grads_bias[i], layer.bias.grad)

    
    # get subsampled Hessian
    u.seed_random(1)
    model = u.SimpleFullyConnected(d, nonlin=False)
    loss_hessian = u.HessianSampledSqrLoss(samples=1)
    output = model(data)

    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True) # 
    model.skip_backward_hooks = True
    i, layer = next(enumerate(model.layers))

    A_t = layer.activations
    Bh_t = layer.backprops_list
    H_approx1 = u.hessian_from_backprops(A_t, Bh_t)

    # use more samples
    u.seed_random(1)
    model = u.SimpleFullyConnected(d, nonlin=False)
    
    loss_hessian = u.HessianSampledSqrLoss(samples=o)
    output = model(data)

    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)
    model.skip_backward_hooks = True
    i, layer = next(enumerate(model.layers))

    A_t = layer.activations
    Bh_t = layer.backprops_list
    H_approx2 = u.hessian_from_backprops(A_t, Bh_t)

    assert abs(u.l2_norm(H)/u.l2_norm(H_approx1)-1) < 0.08, abs(u.l2_norm(H)/u.l2_norm(H_approx1)-1) # 0.0735
    assert abs(u.l2_norm(H)/u.l2_norm(H_approx2)-1) < 0.04, abs(u.l2_norm(H)/u.l2_norm(H_approx2)-1) # 0.0367
    assert u.kl_div_cov(H_approx1, H) < 0.11, u.kl_div_cov(H_approx1, H)   # 0.0673
    assert u.kl_div_cov(H_approx2, H) < 0.03, u.kl_div_cov(H_approx2, H)   # 0.0020




def test_unfold():
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
    X = torch.arange(0, size).reshape(*dims)

    def loss_fn(data):
        err = data.reshape(len(data), -1)
        return torch.sum(err * err) / 2 / len(data)

    layer = model.layers[0]
    output = model(X)
    loss = loss_fn(output)
    loss.backward()

    u.check_close(layer.activations, X)

    assert layer.backprops_list[0].shape == layer.output.shape

    unfold = torch.nn.functional.unfold
    fold = torch.nn.functional.fold
    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (2, 2))
    u.check_close(fold(out_unf, layer.output.shape[2:], (1, 1)), output)


def test_cross_entropy_hessian_tiny():
    u.seed_random(1)

    batch_size = 1
    d = [2, 2]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=True, bias=True)
    model.layers[0].weight.data.copy_(torch.eye(2))

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_hessian = u.HessianExactCrossEntropyLoss()

    data = u.to_logits(torch.tensor([[0.7, 0.3]]))
    targets = torch.tensor([0])

    # get gradient values
    u.clear_backprops(model)
    model.skip_forward_hooks = False
    model.skip_backward_hooks = False
    output = model(data)

    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)
    i = 0
    layer = model.layers[i]
    H, Hbias = u.hessian_from_backprops(layer.activations,
                                        layer.backprops_list,
                                        bias=True)
    model.skip_forward_hooks = True
    model.skip_backward_hooks = True

    # compute Hessian through autograd
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, targets)
    H_autograd = u.hessian(loss, layer.weight)
    u.check_close(H, H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1]))
    Hbias_autograd = u.hessian(loss, layer.bias)
    u.check_close(Hbias, Hbias_autograd)


def test_cross_entropy_hessian_mnist():
    u.seed_random(1)

    data_width = 3
    batch_size = 2
    d = [data_width**2, 10]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=False, bias=True)

    dataset = u.TinyMNIST(dataset_size=batch_size, data_width=data_width, original_targets=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_hessian = u.HessianExactCrossEntropyLoss()

    gl.token_count = 0
    for train_step in range(train_steps):
        data, targets = next(train_iter)

        # get gradient values
        u.clear_backprops(model)
        model.skip_forward_hooks = False
        model.skip_backward_hooks = False
        output = model(data)
        for bval in loss_hessian(output):
            output.backward(bval, retain_graph=True)
        i = 0
        layer = model.layers[i]
        H, Hbias = u.hessian_from_backprops(layer.activations,
                                            layer.backprops_list,
                                            bias=True)
        model.skip_forward_hooks = True
        model.skip_backward_hooks = True

        # compute Hessian through autograd
        model.zero_grad()
        output = model(data)
        loss = loss_fn(output, targets)
        H_autograd = u.hessian(loss, layer.weight).reshape(d[i] * d[i + 1], d[i] * d[i + 1])
        u.check_close(H, H_autograd)
        
        Hbias_autograd = u.hessian(loss, layer.bias)
        u.check_close(Hbias, Hbias_autograd)


def test_conv_grad():
    """Test per-example gradient computation for conv layer."""
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0
    u.seed_random(1)
    N, Xc, Xh, Xw = 3, 2, 3, 7
    dd = [Xc, 2]

    Kh, Kw = 2, 3
    Oh, Ow = Xh - Kh + 1, Xw - Kw + 1
    model = u.SimpleConvolutional(dd, kernel_size=(Kh, Kw), bias=True).double()

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
    output = model(X)
    loss = loss_fn(output)
    loss.backward()

    u.check_equal(layer.activations, X)

    assert layer.backprops_list[0].shape == layer.output.shape
    assert layer.output.shape == (N, dd[1], Oh, Ow)

    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (Kh, Kw))
    assert out_unf.shape == (N, dd[1], Oh * Ow)
    reshaped_bias = layer.bias.reshape(1, dd[1], 1) # (Co,) -> (1, Co, 1)
    out_unf = out_unf + reshaped_bias

    u.check_equal(fold(out_unf, (Oh, Ow), (1, 1)), output)  # two alternative ways of reshaping
    u.check_equal(out_unf.view(N, dd[1], Oh, Ow), output)

    assert unfold(layer.activations, (Kh, Kw)).shape == (N, Xc * Kh * Kw, Oh * Ow)
    assert layer.backprops_list[0].shape == (N, dd[1], Oh, Ow)


    grads_bias = layer.backprops_list[0].sum(dim=(2, 3)) * N
    mean_grad_bias = grads_bias.sum(dim=0) / N
    u.check_equal(mean_grad_bias, layer.bias.grad)

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

    u.check_equal(mean_grad, layer.weight.grad)

    # compute per-example gradients using autograd, compare against manual computation
    for i in range(N):
        u.clear_backprops(model)
        output = model(X[i:i + 1, ...])
        loss = loss_fn(output)
        loss.backward()
        u.check_equal(grads[i], layer.weight.grad)
        u.check_equal(grads_bias[i], layer.bias.grad)


# def test_conv_hessian():
#     """Test per-example gradient computation for conv layer."""
#     u.seed_random(1)
#     N, Xc, Xh, Xw = 3, 2, 3, 7
#     dd = [Xc, 2]
#
#     Kh, Kw = 2, 3
#     Oh, Ow = Xh - Kh + 1, Xw - Kw + 1
#     model = u.ReshapedConvolutional(dd, kernel_size=(Kh, Kw), bias=True)
#
#     weight_buffer = model.layers[0].weight.data
#
#     # output channels, input channels, height, width
#     assert weight_buffer.shape == (dd[1], dd[0], Kh, Kw)
#
#     input_dims = N, Xc, Xh, Xw
#     size = np.prod(input_dims)
#
#     layer = model.layers[0]
#     loss_hessian = u.HessianExactSqrLoss()
#
#     data = torch.ones((N, Xc, Xh, Xw))
#     targets = torch.zeros_like(data)
#
#     output = model(data)
#     for bval in loss_hessian(output):
#         output.backward(bval, retain_graph=True)
#
#     model.skip_backward_hooks = True
#     i, layer = next(enumerate(model.layers))
#
#     assert unfold(layer.activations, (Kh, Kw)).shape == (N, Xc * Kh * Kw, Oh * Ow)
#     assert layer.backprops_list[0].shape == (N, dd[1], Oh, Ow)
#
#
#     # reshape conv activations/backprops into matmul form, then we can
#     # pass these matrices into hessian_from_backprop
#     # we can write
#
#     def matmul_form(bp):
#         """Convert conv backprop matrix into bp suitable for A@bp"""
#         bp = bp.reshape(N, dd[1], Oh * Ow)
#         return bp.transpose(1, 2)
#
#     A_t = unfold(layer.activations, (Kh, Kw))
#     Bh_t = [matmul_form(bp) for bp in layer.backprops_list]
#     assert A_t.shape == (N, dd[0]*Kh*Kw, Oh*Ow)
#     H = u.hessian_from_backprops(A_t, Bh_t)
#     assert Bh_t[0].shape == (N, Oh * Ow, dd[1])
#
#     u.clear_backprops(model)
#     output = model(data)
#     loss = loss_fn(output, targets)
#     model.disable_hooks()
#     H_autograd = u.hessian(loss, layer.weight)
#     u.check_close(H, H_autograd.reshape(H.shape), rtol=1e-4, atol=1e-7)
#
#     # For comparison with autograd, shape needs to be (N, dd[1], dd[0], Kh, Kw)
#     # therefore move output channels to the left, and unmerge remaining shapes
#     grad_unf = grad_unf.transpose(1, 2)
#     grads = grad_unf.reshape((N, dd[1], dd[0], Kh, Kw))
#     mean_grad = torch.sum(grads, dim=0) / N
#
#     u.check_equal(mean_grad, layer.weight.grad)
#
#     # compute per-example gradients using autograd, compare against manual computation
#     for i in range(N):
#         u.clear_backprops(model)
#         output = model(X[i:i + 1, ...])
#         loss = loss_fn(output)
#         loss.backward()
#         u.check_equal(grads[i], layer.weight.grad)
#         u.check_equal(grads_bias[i], layer.bias.grad)
#

if __name__ == '__main__':
    test_hessian()
#    test_conv_hessian()
#    u.run_all_tests(sys.modules[__name__])
    pass

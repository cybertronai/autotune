# Tests that compare manual computation of quantities against PyTorch autograd

import os
import sys
from typing import Any, Dict, Callable

import globals as gl
import torch
import torch.nn as nn
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
    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    model = u.SimpleNet([d1, d2, d3], nonlin=True)

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
    """Newton method towards minimum"""

    image_size = 3
    batch_size = 64
    dataset = u.TinyMNIST('/tmp', download=True, data_width=image_size, targets_width=image_size, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d = image_size ** 2   # hidden layer size
    u.seed_random(1)
    model = u.SimpleNet([d, d])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    def unvec(x):
        return u.unvec(x, d)

    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        if i > 0:
            assert loss < 1e-9

        loss.backward()
        W = model.layers[0].weight
        grad = u.vec(W.grad)

        loss = loss_fn(model(data), targets)
        H = u.hessian(loss, W)
        H = H.transpose(0, 1).transpose(2, 3).reshape(d**2, d**2)

        W1 = unvec(u.vec(W) - u.pinv(H) @ grad)
        W.data.copy_(W1)


def autoencoder_newton_transposed_test():
    """Newton method towards minimum, without transposing Hessian"""

    image_size = 3
    batch_size = 64
    dataset = u.TinyMNIST('/tmp', download=True, data_width=image_size, targets_width=image_size, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d = image_size ** 2   # hidden layer size
    u.seed_random(1)
    model = u.SimpleNet([d, d])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    def untvec(x):
        return u.untvec(x, d)

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
        H = H.reshape(d**2, d**2)

        W1 = untvec(u.tvec(W) - grad @ u.pinv(H))
        W.data.copy_(W1)


def manual_linear_hessian_test():
    u.seed_random(1)

    data_width = 3
    targets_width = 2
    batch_size = 3
    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d1 = data_width ** 2   # hidden layer size, visible size, output size
    d2 = targets_width ** 2   # hidden layer size, visible size, output size
    n = batch_size
    model = u.SimpleNet([d1, d2])
    layer = model.layers[0]
    W = model.layers[0].weight

    skip_hooks = False

    def capture_activations(module, input, _output):
        if skip_hooks:
            return
        assert not hasattr(module, 'activations'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(input) == 1, "this works for single input layers only"
        setattr(module, "activations", input[0].detach())

    def capture_backprops(module: nn.Module, _input, output):
        if skip_hooks:
            return
        assert not hasattr(module, 'backprops'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(output) == 1, "this works for single variable layers only"
        setattr(module, "backprops", output[0])

    layer.register_forward_hook(capture_activations)
    layer.register_backward_hook(capture_backprops)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    #    def unvec(x): return u.unvec(x, d)

    # Gradient
    data, targets = next(iter(trainloader))
    loss = loss_fn(model(data), targets)
    loss.backward()
    A = layer.activations.t()
    assert A.shape == (d1, n)

    # add factor of n here because backprop computes loss averaged over batch, while we need per-example loss backprop
    B = layer.backprops.t() * n
    assert B.shape == (d2, n)
    u.check_close(B @ A.t() / n, W.grad)

    # Hessian
    skip_hooks = True
    loss = loss_fn(model(data), targets)
    H = u.hessian(loss, W)
    H = H.transpose(0, 1).transpose(2, 3).reshape(d1 * d2, d1 * d2)

    # compute B matrices
    Bs_t = []   # one matrix per class, storing backprops for current layer
    skip_hooks = False
    id_mat = torch.eye(d2)
    for out_idx in range(d2):
        u.zero_grad(model)
        output = model(data)
        _loss = loss_fn(output, targets)

        ei = id_mat[out_idx]
        bval = torch.stack([ei]*batch_size)
        output.backward(bval)
        Bs_t.append(layer.backprops)

    A_t = layer.activations

    # batch output Jacobian, each row corresponds to example,output pair
    Amat = torch.cat([A_t]*d2, dim=0)
    Bmat = torch.cat(Bs_t, dim=0)
    Jb = u.khatri_rao_t(Amat, Bmat)
    H2 = Jb.t() @ Jb / n

    u.check_close(H, H2)


def manual_nonlinear_hessian_test():
    u.seed_random(1)

    data_width = 4
    targets_width = 2

    batch_size = 5
    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    n = batch_size
    model = u.SimpleNet([d1, d2, d3], nonlin=True)
    layer = model.layers[0]
    W = model.layers[0].weight

    skip_hooks = False

    def capture_activations(module, input, _output):
        if skip_hooks:
            return
        assert not hasattr(module, 'activations'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(input) == 1, "this works for single input layers only"
        setattr(module, "activations", input[0].detach())

    def capture_backprops(module: nn.Module, _input, output):
        if skip_hooks:
            return
        assert not hasattr(module, 'backprops'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(output) == 1, "this works for single variable layers only"
        setattr(module, "backprops", output[0])

    layer.register_forward_hook(capture_activations)
    layer.register_backward_hook(capture_backprops)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    #    def unvec(x): return u.unvec(x, d)

    # Gradient
    data, targets = next(iter(trainloader))
    loss = loss_fn(model(data), targets)
    loss.backward()
    A = layer.activations.t()
    assert A.shape == (d1, n)

    # add factor of n here because backprop computes loss averaged over batch, while we need per-example loss backprop
    B = layer.backprops.t() * n
    assert B.shape == (d2, n)
    u.check_close(B @ A.t() / n, W.grad)

    # Hessian
    skip_hooks = True
    loss = loss_fn(model(data), targets)
    H_autograd = u.hessian(loss, W)
    H_autograd_t = H_autograd.transpose(0, 1).transpose(2, 3).reshape(d1 * d2, d1 * d2)

    # compute B matrices
    Bs_t = []   # one matrix per class, storing backprops for current layer
    skip_hooks = False
    id_mat = torch.eye(o)
    for out_idx in range(o):
        u.zero_grad(model)
        output = model(data)
        _loss = loss_fn(output, targets)

        ei = id_mat[out_idx]
        bval = torch.stack([ei]*batch_size)
        output.backward(bval)
        Bs_t.append(layer.backprops)

    A_t = layer.activations

    # batch output Jacobian, each row corresponds to example,output pair
    Amat = torch.cat([A_t]*o, dim=0)
    Bmat = torch.cat(Bs_t, dim=0)
    assert Amat.shape == (n*o, d1)
    assert Bmat.shape == (n*o, d2)
    Jb = u.khatri_rao_t(Amat, Bmat)
    H_manual = Jb.t() @ Jb / n
    u.check_close(H_manual, H_autograd_t)

    # we can recover the same row-vectorized order as PyTorch autograd by swapping order in Khatri-Rao
    Jb2 = u.khatri_rao_t(Bmat, Amat)
    H_manual2 = Jb2.t() @ Jb2 / n
    u.check_close(H_manual2, H_autograd.reshape(d1 * d2, d1 * d2))


def log_scalars(metrics: Dict[str, Any]) -> None:
    # TODO(y): move out into util.py
    for tag in metrics:
        gl.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=gl.token_count)


# TODO: rename
def autoencoder_training_test():
    log_wandb = False
    autograd_check = True

    root_logdir = '/temp/autoencoder_test/run'
    count = 0
    while os.path.exists(f"{root_logdir}{count:02d}"):
        count += 1
    logdir = f"{root_logdir}{count:02d}"

    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)

    batch_size = 5
    u.seed_random(1)

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        if log_wandb:
            wandb.init(project='test-graphs_test', name=run_name)
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
    model = u.SimpleNet(d, nonlin=True)
    train_steps = 3

    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size*train_steps)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    # TODO: use values from util
    def capture_activations(module, input, _output):
        if skip_forward_hooks:
            return
        assert gl.backward_idx == 0   # no need to forward-prop on Hessian computation
        assert not hasattr(module, 'activations'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(input) == 1, "this works for single input layers only"
        setattr(module, "activations", input[0].detach())

    def capture_backprops(module: nn.Module, _input, output):
        if skip_backward_hooks:
            return
        assert len(output) == 1, "this works for single variable layers only"
        if gl.backward_idx == 0:
            assert not hasattr(module, 'backprops'), "Seeing results of previous autograd, call util.zero_grad to clear"
            setattr(module, 'backprops', [])
        assert gl.backward_idx == len(module.backprops)
        module.backprops.append(output[0])

    def save_grad(param: nn.Parameter) -> Callable[[torch.Tensor], None]:
        """Hook to save gradient into 'param.saved_grad', so it can be accessed after model.zero_grad(). Only stores gradient
        if the value has not been set, call util.zero_grad to clear it."""
        def save_grad_fn(grad):
            if not hasattr(param, 'saved_grad'):
                setattr(param, 'saved_grad', grad)
        return save_grad_fn

    for layer in model.layers:
        layer.register_forward_hook(capture_activations)
        layer.register_backward_hook(capture_backprops)
        layer.weight.register_hook(save_grad(layer.weight))

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    gl.token_count = 0
    for train_step in range(train_steps):
        data, targets = next(train_iter)
        skip_forward_hooks = False
        skip_backward_hooks = False

        # get gradient values
        gl.backward_idx = 0
        u.zero_grad(model)
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward(retain_graph=True)

        # get Hessian values
        skip_forward_hooks = True
        id_mat = torch.eye(o)
        s = AttrDefault(str, {})
        # o = 0
        for out_idx in range(o):
            model.zero_grad()
            # backprop to get section of batch output jacobian for output at position out_idx
            output = model(data)  # opt: using autograd.grad means I don't have to zero_grad
            ei = id_mat[out_idx]
            bval = torch.stack([ei] * batch_size)
            gl.backward_idx = out_idx+1
            output.backward(bval)
        skip_backward_hooks = True

        for (i, layer) in enumerate(model.layers):

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops[0] * n
            assert B_t.shape == (n, d[i+1])

            G = u.khatri_rao_t(B_t, A_t)
            assert G.shape == (n, d[i]*d[i+1])

            # average gradient
            g = G.sum(dim=0, keepdim=True) / n
            assert g.shape == (1, d[i]*d[i+1])

            if autograd_check:
                u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
                u.check_close(g.reshape(d[i+1], d[i]), layer.weight.saved_grad)

            # empirical Fisher
            efisher = G.t() @ G / n
            _sigma = efisher - g.t() @ g

            #############################
            # Hessian stats
            #############################
            A_t = layer.activations
            Bh_t = [layer.backprops[out_idx+1] for out_idx in range(o)]
            Amat_t = torch.cat([A_t] * o, dim=0)  # todo: can instead replace with a khatri-rao loop
            Bmat_t = torch.cat(Bh_t, dim=0)

            assert Amat_t.shape == (n*o, d[i])
            assert Bmat_t.shape == (n*o, d[i+1])

            # hessian in in row-vectorized layout instead of usual column vectorized, for easy comparison with PyTorch autograd
            Jb = u.khatri_rao_t(Bmat_t, Amat_t)   # batch Jacobian
            H = Jb.t() @ Jb / n

            if autograd_check:
                model.zero_grad()
                output = model(data)  # opt: using autograd.grad means I don't have to zero_grad
                loss = loss_fn(output, targets)
                H_autograd = u.hessian(loss, layer.weight)
                u.check_close(H, H_autograd.reshape(d[i] * d[i+1], d[i] * d[i+1]))
                print("Hessian check passed")

            log_scalars(u.nest_stats(layer.name, s))


def cross_entropy_test():
    data_width = 4
    targets_width = 2
    batch_size = 5
    gl.backward_idx = 0

    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    n = batch_size
    d = [d1, d2, d3]
    model: u.SimpleNet = u.SimpleNet(d, nonlin=True)
    train_steps = 3

    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size * train_steps)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(train_loader)

    for layer in model.layers:
        layer.register_forward_hook(u.capture_activations)
        layer.register_backward_hook(u.capture_backprops)
        layer.weight.register_hook(u.save_grad(layer.weight))

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    gl.token_count = 0
    for train_step in range(train_steps):
        data, targets = next(train_iter)
        gl.skip_forward_hooks = False
        gl.skip_backward_hooks = False

        # get gradient values
        gl.backward_idx = 0
        u.zero_grad(model)
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward(retain_graph=True)

        # get Hessian values
        gl.skip_forward_hooks = True
        id_mat = torch.eye(o)
        s = AttrDefault(str, {})
        # o = 0
        for out_idx in range(o):
            model.zero_grad()
            # backprop to get section of batch output jacobian for output at position out_idx
            output = model(data)
            ei = id_mat[out_idx]
            bval = torch.stack([ei] * batch_size)
            gl.backward_idx = out_idx + 1
            output.backward(bval)

        gl.skip_backward_hooks = True

        for (i, layer) in enumerate(model.layers):

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops[0] * n
            assert B_t.shape == (n, d[i + 1])

            G = u.khatri_rao_t(B_t, A_t)
            assert G.shape == (n, d[i] * d[i + 1])

            # average gradient
            g = G.sum(dim=0, keepdim=True) / n
            assert g.shape == (1, d[i] * d[i + 1])

            u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
            u.check_close(g.reshape(d[i + 1], d[i]), layer.weight.saved_grad)

            # empirical Fisher
            efisher = G.t() @ G / n
            _sigma = efisher - g.t() @ g

            #############################
            # Hessian stats
            #############################
            A_t = layer.activations
            Bh_t = [layer.backprops[out_idx + 1] for out_idx in range(o)]
            Amat_t = torch.cat([A_t] * o, dim=0)
            Bmat_t = torch.cat(Bh_t, dim=0)

            assert Amat_t.shape == (n * o, d[i])
            assert Bmat_t.shape == (n * o, d[i + 1])

            # hessian in in row-vectorized layout instead of usual column vectorized, for easy comparison with PyTorch autograd
            Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch Jacobian
            H = Jb.t() @ Jb / n

            model.zero_grad()
            output = model(data)  # opt: using autograd.grad means I don't have to zero_grad
            loss = loss_fn(output, targets)
            H_autograd = u.hessian(loss, layer.weight)
            u.check_close(H, H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1]))
            print("Hessian check passed")


def unfold_test():
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0

    N, Xc, Xh, Xw = 1, 2, 3, 3
    model = u.SimpleConv([Xc, 2])

    weight_buffer = model.layers[0].weight.data
    weight_buffer.copy_(torch.ones_like(weight_buffer))
    dims = N, Xc, Xh, Xw
    
    size = np.prod(dims)
    X = torch.range(0, size-1).reshape(*dims)

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
    
    assert layer.backprops[0].shape == layer.output.shape

    unfold=torch.nn.functional.unfold
    fold=torch.nn.functional.fold
    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (2, 2))
    u.check_close(fold(out_unf, layer.output.shape[2:], (1, 1)), output)

    print("activations check passed")


# noinspection PyUnresolvedReferences
def conv_grad_test():
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0
    N, Xc, Xh, Xw = 1, 2, 3, 3
    dd = [Xc, 2]
    model: u.SimpleConv = u.SimpleConv(dd)
    Kh, Kw = 2, 2
    Oh, Ow = Xh-Kh+1, Xw-Kw+1

    weight_buffer = model.layers[0].weight.data

    assert weight_buffer.shape == (dd[1], dd[0], Kh, Kw)

    # first output channel=1's, second channel=2's
    weight_buffer[0, :, :, :].copy_(torch.ones_like(weight_buffer[0, :, :, :]))
    weight_buffer[1, :, :, :].copy_(2*torch.ones_like(weight_buffer[1, :, :, :]))
    
    dims = N, Xc, Xh, Xw
    
    size = np.prod(dims)
    X = torch.range(0, size-1).reshape(*dims)

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
    
    assert layer.backprops[0].shape == layer.output.shape

    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (2, 2))
    u.check_close(fold(out_unf, layer.output.shape[2:], (1, 1)), output)

    assert unfold(layer.activations, (Oh, Ow)).shape == (N, Xc*Kh*Kw, Oh*Ow)
    assert layer.backprops[0].shape == (N, dd[1], Oh, Ow)

    # make patches be the inner dimension
    bp = layer.backprops[0]
    bp = bp.reshape(N, dd[1], Oh*Ow)
    bp = bp.transpose(1, 2)

    print('backprops')
    print(bp)
    grad_unf = unfold(layer.activations, (Oh, Ow)) @ bp
    assert grad_unf.shape == (N, dd[0]*Kh*Kw, dd[1]) # need (dd[1], dd[0], Kh, Kw)
    grad_unf = grad_unf.transpose(1, 2)
    grads = grad_unf.reshape((N, dd[1], dd[0], Kh, Kw))
    assert N==1, "currently only works for N=1"
    print('predicted')
    print(grads[0])
    print('actual')
    print(layer.weight.grad)
    print(torch.max(grads[0]-layer.weight.grad))
    u.check_equal(grads[0], layer.weight.grad)
    print("grad check passed")


def conv_multiexample_test():
    gl.skip_backward_hooks = False
    gl.skip_forward_hooks = False
    gl.backward_idx = 0
    u.seed_random(1)
    N, Xc, Xh, Xw = 3, 2, 3, 7
    dd = [Xc, 2]

    Kh, Kw = 2, 3
    Oh, Ow = Xh-Kh+1, Xw-Kw+1
    model = u.SimpleConv(dd, kernel_size=(Kh, Kw)).double()

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
    
    assert layer.backprops[0].shape == layer.output.shape
    assert layer.output.shape == (N, dd[1], Oh, Ow)

    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (Kh, Kw))
    u.check_equal(fold(out_unf, (Oh, Ow), (1, 1)), output)
    u.check_equal(out_unf.view(N, dd[1], Oh, Ow), output)
    
    #    print(unfold(layer.activations, (Kh, Kw)))
    assert unfold(layer.activations, (Kh, Kw)).shape == (N, Xc*Kh*Kw, Oh*Ow)
    assert layer.backprops[0].shape == (N, dd[1], Oh, Ow)

    bp = layer.backprops[0] * N   # remove factor of N applied during loss batch averaging

    # merge patches into single dimension, move patches to be the inner dimension, output channels the rightmost dimension
    # since we multiplying on the left. TODO(y) try einsum for this and benchmark
    bp = bp.reshape(N, dd[1], Oh*Ow)
    bp = bp.transpose(1, 2)

    grad_unf = unfold(layer.activations, (Kh, Kw)) @ bp
    assert grad_unf.shape == (N, dd[0]*Kh*Kw, dd[1])

    # For comparison with autograd, shape needs to be (N, dd[1], dd[0], Kh, Kw)
    # therefore move output channels to the left, and unmerge remaining shapes
    grad_unf = grad_unf.transpose(1, 2)
    grads = grad_unf.reshape((N, dd[1], dd[0], Kh, Kw))
    mean_grad = torch.sum(grads, dim=0)/N

    print(f'grad: {torch.max(abs(mean_grad-layer.weight.grad))}')
    u.check_equal(mean_grad, layer.weight.grad)

    # compute per-example gradients using autograd, compare against manual computation
    for i in range(N):
        u.zero_grad(model)
        output = model(X[i:i+1,...])
        loss = loss_fn(output)
        loss.backward()
        print(f'grad {i}: {torch.max(abs(grads[i]-layer.weight.grad))}')
        u.check_equal(grads[i], layer.weight.grad)


if __name__ == '__main__':
    #    conv_grad_test()
#    conv_multiexample_test()
#    sys.exit()
#    unfold_test()
#    cross_entropy_test()
    #    autoencoder_minimize_test()
    #    autoencoder2_minimize_test()
    #    autoencoder_newton_test()
    #    autoencoder_newton_transposed_test()
    #    manual_linear_hessian_test()
    #    manual_nonlinear_hessian_test()
    #    autoencoder_training_test()
    u.run_all_tests(sys.modules[__name__])

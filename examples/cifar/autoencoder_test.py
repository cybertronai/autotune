import os
import sys
from typing import Any, Dict, Callable
from typing import List

import globals as gl
# import torch
import torch
import torch.nn as nn
import wandb
from attrdict import AttrDefault
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# Test exact Hessian computation

module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path)
import util as u


def autoencoder_minimize_test():
    """Minimize autoencoder for a few steps."""
    data_width = 4
    targets_width = 2
    batch_size = 64
    dataset = u.TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d = data_width ** 2
    model = u.SimpleNet([d, targets_width ** 2])

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
            assert loss > 0.03
        print(loss)
        loss.backward()
        optimizer.step()

    assert loss < 0.015


def autoencoder2_minimize_test():
    """deeper autoencoder."""
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
        print(loss)
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
        print(loss)
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
        print(loss.item())
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
    print(H)

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
    print(H_autograd_t)

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
            Amat_t = torch.cat([A_t] * o, dim=0)
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


if __name__ == '__main__':
    #    autoencoder_minimize_test()
    #    autoencoder2_minimize_test()
    #    autoencoder_newton_test()
    #    autoencoder_newton_transposed_test()
    #    manual_linear_hessian_test()
    #    manual_nonlinear_hessian_test()
    #    autoencoder_training_test()
    u.run_all_tests(sys.modules[__name__])

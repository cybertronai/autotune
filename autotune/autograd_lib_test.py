import sys
from collections import namedtuple, defaultdict

import autograd_lib
import pytest

import util as u

# Test exact Hessian computation

# import torch
from typing import Callable

import torch
import torch.nn as nn

from attrdict import AttrDefault


def simple_model(d, num_layers):
    """Creates simple linear neural network initialized to identity"""
    layers = []
    for i in range(num_layers):
        layer = nn.Linear(d, d, bias=False)
        layer.weight.data.copy_(torch.eye(d))
        layers.append(layer)
    return torch.nn.Sequential(*layers)


def test_hooks():
    d = 1
    model = simple_model(d, num_layers=5)
    autograd_lib.register(model)

    A1, A2, A3 = {}, {}, {}
    x = torch.ones(1, d)

    with autograd_lib.save_activations(A1):
        y = model(2 * x)

    with autograd_lib.save_activations(A2):
        with autograd_lib.save_activations(A3):
            y = model(x)

    B1 = {}
    B2 = {}
    with autograd_lib.extend_backprops(B1):
        y.backward(x, retain_graph=True)

    model[2].weight.requires_grad = False
    for layer in model:
        del layer.weight.grad

    # model.clear_grads()
    with autograd_lib.extend_backprops(B2):
        y.backward(2 * x)

    print(B2.values())
    for layer in model:
        print(layer.weight.grad)

    for layer in model:
        assert A1[layer] == 2 * x
        assert A2[layer] == x
        assert A3[layer] == x
        assert B1[layer] == [x]
        assert B2[layer] == [2 * x]

    autograd_lib.unregister()


def _test_activations_contextmanager():
    d = 5
    model = simple_model(d, num_layers=2)
    autograd_lib.register(model)

    A1, A2, A3 = {}, {}, {}
    x = torch.ones(1, d)

    with autograd_lib.save_activations(A1):
        y = model(x)
        with autograd_lib.save_activations(A2):
            z = model[1](x)

    context_ids = autograd_lib.global_settings.last_captured_activations_contextid
    assert context_ids[model[1]] == context_ids[model[0]] + 1


def _test_backprop():
    d = 1
    model = simple_model(d, num_layers=5)
    autograd_lib.register(model)

    x = torch.ones(2, d)
    y = model(x)

    # make sure buffers get freed, second call will cause a crash
    autograd_lib.backward(y, kind='identity')
    with pytest.raises(RuntimeError, match=r".*retain_graph=True.*"):
        autograd_lib.backward(y, kind='identity')

    y = model(x)
    B = {}
    with autograd_lib.save_backprops(B):
        autograd_lib.backward(y, kind='identity', retain_graph=True)
    u.check_equal(B[model[0]], [x])

    with autograd_lib.save_backprops(B):
        autograd_lib.backward(y, kind='identity', retain_graph=True)
    u.check_equal(B[model[0]], [x, x])

    autograd_lib.unregister()


def test_jacobian():
    # ground truth for unit tests from
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/linear-jacobians-and-hessians.nb

    def init_model(B, X, A):
        """  Initializes the model Y=B'XA
        """
        B, X, A = u.to_pytorches(B, X, A)
        n = A.shape[1]
        d1, d2 = X.shape
        d3 = B.shape[1]

        # Do a test using Linear layers instead of matrix multiplies
        model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([d1, d2, d3], bias=False)
        model.layers[0].weight.data.copy_(X)
        model.layers[1].weight.data.copy_(B.t())

        def eval():
            return model(A.t())

        return eval, model.layers[0].weight

    # default Kronecker rules give result in vec order.
    # A*B=>(B*A)'  gives scalar for vector or scalar jacobian in vecr order
    # For matrix/matrix Jacobian must also switch the first two dimensions

    # matrix variable, scalar output
    torch.set_default_dtype(torch.float64)
    B = torch.tensor([[-4.], [2]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)
    A = torch.tensor([[-1.], [3]])
    d_out, d_in = X.shape
    Y_func, X_var = init_model(B, X, A)
    Y = Y_func()
    u.check_equal(Y, [[-52]])

    J = u.jacobian(Y, X_var)
    assert J.shape == (1, 1, 2, 2)
    J = J.reshape(2, 2)

    u.check_equal(J, u.kron(B, A).T.reshape(d_out, d_in))
    u.check_equal(J, [[4, -12], [-2, 6]])

    # matrix variable, vector output, dvecr Y/dvecr X
    B = [[-4, 3], [2, 6]]
    B, X, A = u.to_pytorches(B, X, A)
    Y_func, X_var = init_model(B, X, A)
    Y = Y_func()
    u.check_equal(Y, [[-52, -81]])
    J = u.jacobian(Y, X_var)
    assert J.shape == (1, 2, 2, 2)
    J1 = u.kron(B, A).T
    assert J1.shape == (2, 4)  # output and input directions are flattened
    u.check_equal(J, J1.reshape(J.shape))
    u.check_equal(J.reshape(J1.shape), J1)

    # matrix variable, matrix output, dvecr Y/dvecX
    A = torch.tensor([[-1., 4], [3, 0]])
    B, X, A = u.to_pytorches(B, X, A)
    Y_func, X_var = init_model(B, X, A)
    Y = Y_func()

    J = u.jacobian(Y, X_var)
    J = J.transpose(0, 1)  # dvecrY/dvecr X -> dvecY/dvecr X
    assert J.shape == (2, 2, 2, 2)

    J1 = u.kron(B, A).T  # this gives order where variable is row vectorized, but output is column vectorized
    assert J1.shape == (4, 4)
    u.check_equal(J, J1.reshape(J.shape))
    u.check_equal(J.reshape(J1.shape), J1)

    # Hessian of matrix variable,  x output
    loss = (Y * Y).sum() / 2
    hess = u.hessian(loss, X_var)
    assert hess.shape == (2, 2, 2, 2)
    hess1 = u.kron(B @ B.t(), A @ A.t())
    assert hess1.shape == (4, 4)
    u.check_equal(hess1.reshape(hess.shape), hess)
    u.check_equal(hess1, hess.reshape(hess1.shape))


def create_toy_model():
    """
    Create model from https://www.wolframcloud.com/obj/yaroslavvb/newton/linear-jacobians-and-hessians.nb
    PyTorch works on transposed representation, hence to obtain Y from notebook, do model(A.T).T
    """

    model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([2, 2, 2], bias=False)
    autograd_lib.register(model)

    A = torch.tensor([[-1., 4], [3, 0]])
    B = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    model.layers[0].weight.data.copy_(X)
    model.layers[1].weight.data.copy_(B.t())
    return A, model


def test_gradient_norms():
    """Per-example gradient norms."""
    u.seed_random(1)
    A, model = create_toy_model()

    activations = {}

    def save_activations(layer, a, _):
        if layer != model.layers[0]:
            return
        activations[layer] = a

    with autograd_lib.module_hook(save_activations):
        Y = model(A.t())
        loss = torch.sum(Y * Y) / 2

    norms = {}

    def compute_norms(layer, _, b):
        if layer != model.layers[0]:
            return
        a = activations[layer]
        del activations[layer]  # release memory kept by activations
        norms[layer] = (a * a).sum(dim=1) * (b * b).sum(dim=1)

    with autograd_lib.module_hook(compute_norms):
        loss.backward()

    u.check_equal(norms[model.layers[0]], [3493250, 9708800])


def test_full_hessian():
    u.seed_random(1)
    A, model = create_toy_model()
    data = A.t()
    #    data = data.repeat(3, 1)
    activations = {}

    hess = defaultdict(float)
    def save_activations(layer, a, _):
        activations[layer] = a

    with autograd_lib.module_hook(save_activations):
        Y = model(A.t())
        loss = torch.sum(Y * Y) / 2

    def compute_hess(layer, _, B):
        A = activations[layer]
        n = A.shape[0]

        di = A.shape[1]
        do = B.shape[1]

        BA = torch.einsum("nl,ni->nli", B, A)
        hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backprop_identity(Y, retain_graph=True)

    # check against autograd
    hess_autograd = u.hessian(loss, model.layers[0].weight)
    hess0 = hess[model.layers[0]]
    u.check_equal(hess_autograd, hess0)

    # check against manual solution
    u.check_equal(hess0.reshape(4, 4), [[425, -75, 170, -30], [-75, 225, -30, 90], [170, -30, 680, -120], [-30, 90, -120, 360]])


def test_full_fisher():
    u.seed_random(1)
    A, model = create_toy_model()

    activations = {}

    def save_activations(layer, a, _):
        if layer != model.layers[0]:
            return
        activations[layer] = a

    with autograd_lib.module_hook(save_activations):
        Y = model(A.t())
        loss = torch.sum(Y * Y) / 2

    fisher = [0]

    def compute_fisher(layer, _, B):
        if layer != model.layers[0]:
            return
        A = activations[layer]
        n = A.shape[0]

        di = A.shape[1]
        do = B.shape[1]

        Jo = torch.einsum("ni,nj->nij", B, A).reshape(n, -1)
        fisher[0] += torch.einsum('ni,nj->ij', Jo, Jo)

    with autograd_lib.module_hook(compute_fisher):
        loss.backward()

    result0 = torch.tensor([[5.383625e+06, -3.675000e+03, 4.846250e+06, -6.195000e+04],
                            [-3.675000e+03, 1.102500e+04, -6.195000e+04, 1.858500e+05],
                            [4.846250e+06, -6.195000e+04, 4.674500e+06, -1.044300e+06],
                            [-6.195000e+04, 1.858500e+05, -1.044300e+06, 3.132900e+06]])
    u.check_close(fisher[0], result0)


def test_full_fisher_multibatch():
    torch.set_default_dtype(torch.float64)
    u.seed_random(1)
    A, model = create_toy_model()

    activations = {}

    def save_activations(layer, a, _):
        if layer != model.layers[0]:
            return
        activations[layer] = a

    fisher = [0]

    def compute_fisher(layer, _, B):
        if layer != model.layers[0]:
            return
        A = activations[layer]
        n = A.shape[0]

        di = A.shape[1]
        do = B.shape[1]

        Jo = torch.einsum("ni,nj->nij", B, A).reshape(n, -1)
        fisher[0] += torch.einsum('ni,nj->ij', Jo, Jo)

    for x in A.t():
        print(x)
        with autograd_lib.module_hook(save_activations):
            y = model(x)
            loss = torch.sum(y * y) / 2

        with autograd_lib.module_hook(compute_fisher):
            loss.backward()

    # result computed using single step forward prop
    result0 = torch.tensor([[5.383625e+06, -3.675000e+03, 4.846250e+06, -6.195000e+04],
                            [-3.675000e+03, 1.102500e+04, -6.195000e+04, 1.858500e+05],
                            [4.846250e+06, -6.195000e+04, 4.674500e+06, -1.044300e+06],
                            [-6.195000e+04, 1.858500e+05, -1.044300e+06, 3.132900e+06]])
    u.check_close(fisher[0], result0)
    # check against autograd
    # hess0 = u.hessian(loss, model.layers[0].weight).reshape([4, 4])
    # u.check_equal(hess[0], hess0)


def test_kfac_hessian():
    A, model = create_toy_model()
    data = A.t()
    data = data.repeat(7, 1)
    n = float(len(data))

    activations = {}
    hess = defaultdict(lambda: AttrDefault(float))
    def save_activations(layer, a, _):
        activations[layer] = a
    def compute_hessian(layer, _, B):
        A = activations[layer]
        hess[layer].AA += torch.einsum("ni,nj->ij", A, A)
        hess[layer].BB += torch.einsum("ni,nj->ij", B, B)

    for x in data:
        with autograd_lib.module_hook(save_activations):
            y = model(x)
            o = y.shape[1]
            loss = torch.sum(y * y) / 2

        with autograd_lib.module_hook(compute_hessian):
            autograd_lib.backprop_identity(y)

    hess0 = hess[model.layers[0]]
    result = u.kron(hess0.BB / n, hess0.AA / o)

    # check result against autograd
    loss = u.least_squares(model(data), aggregation='sum')
    hess0 = u.hessian(loss, model.layers[0].weight).reshape(4, 4)
    u.check_equal(hess0, result)


def test_full_hessian_multibatch():
    A, model = create_toy_model()
    data = A.t()
    data = data.repeat(3, 1)
    n = float(len(data))

    activations = {}
    hess = defaultdict(float)
    def save_activations(layer, a, _):
        activations[layer] = a
    def compute_hessian(layer, _, B):
        A = activations[layer]
        BA = torch.einsum("nl,ni->nli", B, A)
        hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    for x in data:
        with autograd_lib.module_hook(save_activations):
            y = model(x)
            loss = torch.sum(y * y) / 2

        with autograd_lib.module_hook(compute_hessian):
            autograd_lib.backprop_identity(y)

    result = hess[model.layers[0]]

    # check result against autograd
    loss = u.least_squares(model(data), aggregation='sum')
    hess0 = u.hessian(loss, model.layers[0].weight)
    u.check_equal(hess0, result)


def test_diagonal_hessian():
    u.seed_random(1)
    A, model = create_toy_model()

    activations = {}

    def save_activations(layer, a, _):
        if layer != model.layers[0]:
            return
        activations[layer] = a

    with autograd_lib.module_hook(save_activations):
        Y = model(A.t())
        loss = torch.sum(Y * Y) / 2

    hess = [0]

    def compute_hess(layer, _, B):
        if layer != model.layers[0]:
            return
        A = activations[layer]
        hess[0] += torch.einsum("ni,nj->ij", B * B, A * A).reshape(-1)

    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backprop_identity(Y, retain_graph=True)

    # check against autograd
    hess0 = u.hessian(loss, model.layers[0].weight).reshape([4, 4])
    u.check_equal(hess[0], torch.diag(hess0))

    # check against manual solution
    u.check_equal(hess[0], [425., 225., 680., 360.])


def test_full_hessian_xent():
    u.seed_random(1)
    torch.set_default_dtype(torch.float64)

    batch_size = 1
    d = [2, 2]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=True, bias=True)
    model.layers[0].weight.data.copy_(torch.eye(2))
    autograd_lib.register(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    data = u.to_logits(torch.tensor([[0.7, 0.3]]))
    targets = torch.tensor([0])

    data = data.repeat([3, 1])
    targets = targets.repeat([3])
    n = len(data)

    activations = {}
    hess = defaultdict(float)
    def save_activations(layer, a, _):
        activations[layer] = a

    with autograd_lib.module_hook(save_activations):
        Y = model(data)
        loss = loss_fn(Y, targets)

    def compute_hess(layer, _, B):
        A = activations[layer]
        BA = torch.einsum("nl,ni->nli", B, A)
        hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(Y, loss='CrossEntropy', retain_graph=True)

    # check against autograd
    # 0.1459
    hess_autograd = u.hessian(loss, model.layers[0].weight)
    hess0 = hess[model.layers[0]] / n
    u.check_equal(hess_autograd, hess0)


def test_full_hessian_xent_multibatch():
    u.seed_random(1)
    torch.set_default_dtype(torch.float64)

    batch_size = 1
    d = [2, 2]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected2(d, nonlin=True, bias=True)
    model.layers[0].weight.data.copy_(torch.eye(2))
    autograd_lib.register(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    data = u.to_logits(torch.tensor([[0.7, 0.3]]))
    targets = torch.tensor([0])

    data = data.repeat([3, 1])
    targets = targets.repeat([3])
    n = len(data)

    activations = {}
    hess = defaultdict(float)
    def save_activations(layer, a, _):
        activations[layer] = a

    for i in range(n):
        with autograd_lib.module_hook(save_activations):
            data_batch = data[i: i+1]
            targets_batch = targets[i: i+1]
            Y = model(data_batch)
            loss = loss_fn(Y, targets_batch)

        def compute_hess(layer, _, B):
            A = activations[layer]
            BA = torch.einsum("nl,ni->nli", B, A)
            hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(Y, loss='CrossEntropy')

    # check against autograd
    # 0.1459
    Y = model(data)
    loss = loss_fn(Y, targets)
    hess_autograd = u.hessian(loss, model.layers[0].weight)
    hess0 = hess[model.layers[0]] / n
    u.check_equal(hess_autograd, hess0)


def test_full_hessian_xent_kfac():
    u.seed_random(1)
    torch.set_default_dtype(torch.float64)

    batch_size = 1
    d = [2, 2]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected2(d, nonlin=True, bias=True)
    model.layers[0].weight.data.copy_(torch.eye(2))
    autograd_lib.register(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    data = u.to_logits(torch.tensor([[0.7, 0.3]]))
    targets = torch.tensor([0])

    data = data.repeat([3, 1])
    targets = targets.repeat([3])
    n = len(data)

    activations = {}
    hess = defaultdict(lambda: AttrDefault(float))

    for i in range(n):
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            data_batch = data[i: i+1]
            targets_batch = targets[i: i+1]
            Y = model(data_batch)
            o = Y.shape[1]
            loss = loss_fn(Y, targets_batch)

        def compute_hess(layer, _, B):
            A = activations[layer]
            hess[layer].AA += torch.einsum("ni,nj->ij", A, A)
            hess[layer].BB += torch.einsum("ni,nj->ij", B, B)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(Y, loss='CrossEntropy')

    # expand
    hess_factored = hess[model.layers[0]]
    hess0 = torch.einsum('kl,ij->kilj', hess_factored.BB / n, hess_factored.AA / o)  # hess for sum loss
    hess0 /= n  # hess for mean loss


    # check against autograd
    # 0.1459
    Y = model(data)
    loss = loss_fn(Y, targets)
    hess_autograd = u.hessian(loss, model.layers[0].weight)
    u.check_equal(hess_autograd, hess0)

    # check diagonal hessian
    diag_autograd = torch.einsum('lili->li', hess_autograd)
    diag_kfac = torch.einsum('ll,ii->li', hess_factored.BB / n, hess_factored.AA / o / n)
    u.check_close(diag_autograd,  diag_kfac)



def test_full_hessian_xent_kfac2():
    """Test with uneven layers."""
    u.seed_random(1)
    torch.set_default_dtype(torch.float64)

    batch_size = 1
    d = [3, 2]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected2(d, nonlin=True, bias=False)
    autograd_lib.register(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    data = u.to_logits(torch.tensor([[0.7, 0.2, 0.1]]))
    targets = torch.tensor([0])

    data = data.repeat([3, 1])
    targets = targets.repeat([3])
    n = len(data)

    activations = {}
    hess = defaultdict(lambda: AttrDefault(float))

    for i in range(n):
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            data_batch = data[i: i+1]
            targets_batch = targets[i: i+1]
            Y = model(data_batch)
            o = Y.shape[1]
            loss = loss_fn(Y, targets_batch)

        def compute_hess(layer, _, B):
            A = activations[layer]
            hess[layer].AA += torch.einsum("ni,nj->ij", A, A)
            hess[layer].BB += torch.einsum("ni,nj->ij", B, B)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(Y, loss='CrossEntropy')

    # expand
    hess_factored = hess[model.layers[0]]
    hess0 = torch.einsum('kl,ij->kilj', hess_factored.BB / n, hess_factored.AA / o)  # hess for sum loss
    hess0 /= n  # hess for mean loss

    # check against autograd
    # 0.1459
    Y = model(data)
    loss = loss_fn(Y, targets)
    hess_autograd = u.hessian(loss, model.layers[0].weight)
    u.check_equal(hess_autograd, hess0)


def test_full_hessian_xent_mnist():
    u.seed_random(1)

    data_width = 3
    batch_size = 2
    d = [data_width**2, 10]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected2(d, nonlin=False, bias=True)
    autograd_lib.register(model)
    dataset = u.TinyMNIST(dataset_size=batch_size, data_width=data_width, original_targets=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()

    hess = defaultdict(float)
    for train_step in range(train_steps):
        data, targets = next(train_iter)

        activations = {}
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, targets)

        def compute_hess(layer, _, B):
            A = activations[layer]
            BA = torch.einsum("nl,ni->nli", B, A)
            hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(output, loss='CrossEntropy', retain_graph=True)

        # compute Hessian through autograd
        H_autograd = u.hessian(loss, model.layers[0].weight)
        u.check_close(hess[model.layers[0]] / n, H_autograd)


def test_full_hessian_xent_mnist_multilayer():
    """Test regular and diagonal hessian computation."""
    u.seed_random(1)

    data_width = 3
    batch_size = 2
    d = [data_width**2, 6, 10]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected2(d, nonlin=False, bias=True)
    autograd_lib.register(model)
    dataset = u.TinyMNIST(dataset_size=batch_size, data_width=data_width, original_targets=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()

    hess = defaultdict(float)
    hess_diag = defaultdict(float)
    for train_step in range(train_steps):
        data, targets = next(train_iter)

        activations = {}
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, targets)

        def compute_hess(layer, _, B):
            A = activations[layer]
            BA = torch.einsum("nl,ni->nli", B, A)
            hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)
            hess_diag[layer] += torch.einsum("ni,nj->ij", B * B, A * A)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(output, loss='CrossEntropy', retain_graph=True)

        # compute Hessian through autograd
        H_autograd = u.hessian(loss, model.layers[0].weight)
        u.check_close(hess[model.layers[0]] / batch_size, H_autograd)
        diag_autograd = torch.einsum('lili->li', H_autograd)
        u.check_close(diag_autograd, hess_diag[model.layers[0]]/batch_size)

        H_autograd = u.hessian(loss, model.layers[1].weight)
        u.check_close(hess[model.layers[1]] / batch_size, H_autograd)
        diag_autograd = torch.einsum('lili->li', H_autograd)
        u.check_close(diag_autograd, hess_diag[model.layers[1]]/batch_size)


def test_kfac_hessian_xent_mnist():
    u.seed_random(1)

    data_width = 3
    batch_size = 2
    d = [data_width**2, 10]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model: u.SimpleModel = u.SimpleFullyConnected2(d, nonlin=False, bias=True)
    autograd_lib.register(model)
    dataset = u.TinyMNIST(dataset_size=batch_size, data_width=data_width, original_targets=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()

    activations = {}
    hess = defaultdict(lambda: AttrDefault(float))
    for train_step in range(train_steps):
        data, targets = next(train_iter)

        activations = {}
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, targets)

        def compute_hess(layer, _, B):
            A = activations[layer]
            hess[layer].AA += torch.einsum("ni,nj->ij", A, A)
            hess[layer].BB += torch.einsum("ni,nj->ij", B, B)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(output, loss='CrossEntropy', retain_graph=True)

        hess_factored = hess[model.layers[0]]
        hess0 = torch.einsum('kl,ij->kilj', hess_factored.BB / n, hess_factored.AA / o)  # hess for sum loss
        hess0 /= n  # hess for mean loss

        # compute Hessian through autograd
        H_autograd = u.hessian(loss, model.layers[0].weight)
        rel_error = torch.norm((hess0-H_autograd).flatten())/torch.norm(H_autograd.flatten())
        assert rel_error < 0.01   # 0.0057


if __name__ == '__main__':
    # test_gradient_norms()
    # test_full_hessian()
    # test_diagonal_hessian()
    # test_full_fisher()
    # test_full_fisher_multibatch()
    #  test_full_hessian_multibatch()
    #  test_kfac_hessian()
    # test_full_hessian_xent()
    #    test_full_hessian_xent_multibatch()
    test_full_hessian_xent_kfac()
    test_full_hessian_xent_kfac2()
    # test_full_hessian_xent_mnist()
    test_full_hessian_xent_mnist_multilayer()
    test_kfac_hessian_xent_mnist()
    # test_hooks()
    # test_activations_contextmanager()
    # test_jacobian()
    # test_backprop()
    # u.run_all_tests(sys.modules[__name__])
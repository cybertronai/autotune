import sys

import autograd_lib
import pytest

import util as u


# Test exact Hessian computation

# import torch
from typing import Callable

import torch
import torch.nn as nn


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
        y = model(2*x)

    with autograd_lib.save_activations(A2):
        with autograd_lib.save_activations(A3):
            y = model(x)

    B1 = {}
    B2 = {}
    with autograd_lib.save_backprops(B1):
        y.backward(x, retain_graph=True)

    with autograd_lib.save_backprops(B2):
        y.backward(2*x)

    for layer in model:
        assert A1[layer] == 2*x
        assert A2[layer] == x
        assert A3[layer] == x
        assert B1[layer] == [x]
        assert B2[layer] == [2*x]

    autograd_lib.unregister()


def test_backprop():
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
    assert J1.shape == (2, 4)    # output and input directions are flattened
    u.check_equal(J, J1.reshape(J.shape))
    u.check_equal(J.reshape(J1.shape), J1)

    # matrix variable, matrix output, dvecr Y/dvecX
    A = torch.tensor([[-1., 4], [3, 0]])
    B, X, A = u.to_pytorches(B, X, A)
    Y_func, X_var = init_model(B, X, A)
    Y = Y_func()

    J = u.jacobian(Y, X_var)
    J = J.transpose(0, 1)   # dvecrY/dvecr X -> dvecY/dvecr X
    assert J.shape == (2, 2, 2, 2)

    J1 = u.kron(B, A).T  # this gives order where variable is row vectorized, but output is column vectorized
    assert J1.shape == (4, 4)
    u.check_equal(J, J1.reshape(J.shape))
    u.check_equal(J.reshape(J1.shape), J1)

    # Hessian of matrix variable,  x output
    loss = (Y*Y).sum()/2
    hess = u.hessian(loss, X_var)
    assert hess.shape == (2, 2, 2, 2)
    hess1 = u.kron(B@B.t(), A@A.t())
    assert hess1.shape == (4, 4)
    u.check_equal(hess1.reshape(hess.shape), hess)
    u.check_equal(hess1, hess.reshape(hess1.shape))


if __name__ == '__main__':
    # test_hooks()
    test_jacobian()
    # test_backprop()
    # u.run_all_tests(sys.modules[__name__])

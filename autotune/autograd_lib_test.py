import autograd_lib

import util as u


# Test exact Hessian computation

# import torch
from typing import Callable

import torch
import torch.nn as nn


def test_hooks():

    def simple_model(d, num_layers):
        """Creates linear neural network initialized to identity"""
        layers = []
        for i in range(num_layers):
            layer = nn.Linear(d, d, bias=False)
            layer.weight.data.copy_(torch.eye(d))
            layers.append(layer)
        return torch.nn.Sequential(*layers)

    d = 1
    model = simple_model(d, num_layers=5)
    autograd_lib.register(model)

    A1 = autograd_lib.ModuleDict()
    A2 = autograd_lib.ModuleDict()
    A3 = autograd_lib.ModuleDict()

    x = torch.ones(1, d)

    with autograd_lib.save_activations(A3):
        y = model(2*x)

    with autograd_lib.save_activations(A1):
        with autograd_lib.save_activations(A2):
            y = model(x)

    B1 = autograd_lib.ModuleDict()
    B2 = autograd_lib.ModuleDict()
    with autograd_lib.save_backprops(B1):
        y.backward(x, retain_graph=True)

    with autograd_lib.save_backprops(B2):
        y.backward(2*x)

    for layer in model:
        assert A1[layer] == x
        assert A2[layer] == x
        assert A3[layer] == 2*x
        assert B1[layer] == [x]
        assert B2[layer] == [2*x]


def _test_hessian():
    torch.set_default_dtype(torch.float64)
    Amat = torch.tensor([[-1., 4], [3, 0]])
    Bmat = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    Y = Bmat.t() @ X @ Amat
    u.check_equal(Y, [[-52, 64], [-81, -108]])
    loss = torch.sum(Y * Y) / 2
    hess0 = u.hessian(loss, X).reshape([4, 4])
    hess1 = u.Kron(Bmat @ Bmat.t(), Amat @ Amat.t())

    u.check_equal(loss, 12512.5)
    u.check_equal(hess1, hess0)

    # Do a test using Linear layers instead of matrix multiplies
    model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([2, 2, 2], bias=False)
    model.layers[0].weight.data.copy_(X)

    # Linear layers are equivalent to multiplying on the right, treating data matrices as rows of datapoints
    # Transpose to match previous results
    u.check_equal(model.layers[0](Amat.t()).t(), [[5, -20], [-16, -8]])  # XA = (A'X0)'

    model.layers[1].weight.data.copy_(Bmat.t())
    u.check_equal(model(Amat.t()).t(), Y)

    A, B = {}, {}
    with autograd_lib.save_activations(A):
        Y = model(Amat.t()).t()    # transpose to data-dimension=columns
        loss = torch.sum(Y * Y) / 2

    with autograd_lib.save_backprops(B):
        autograd_lib.backprop(Y, "identity")

    print(B)
    print(Bmat)


if __name__ == '__main__':
    test_hooks()
    # test_hessian()
    # u.run_all_tests(sys.modules[__name__])

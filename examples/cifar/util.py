# Take simple example, plot per-layer stats over time
import inspect
import math
import random
import sys
import time
from typing import Dict

import numpy as np
import scipy
import torch


import torch.nn as nn

import torchvision.datasets as datasets
from PIL import Image

import globals as gl

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



def v2c(vec):
    """Convert vector to column matrix."""
    assert len(vec.shape) == 1
    return torch.unsqueeze(vec, 1)


def v2c_np(vec):
    """Convert vector to column matrix."""
    assert len(vec.shape) == 1
    return np.expand_dims(vec, 1)


def v2r(vec: torch.Tensor) -> torch.Tensor:
    """Converts rank-1 tensor to row matrix"""
    assert len(vec.shape) == 1
    return vec.unsqueeze(0)


def c2v(col: torch.Tensor) -> torch.Tensor:
    """Convert vector into row matrix."""
    assert len(col.shape) == 2
    assert col.shape[1] == 1
    return torch.reshape(col, [-1])


def vec(mat):
    """vec operator, stack columns of the matrix into single column matrix."""
    assert len(mat.shape) == 2
    return mat.t().reshape(-1, 1)


def vec_test():
    mat = torch.tensor([[1, 3, 5], [2, 4, 6]])
    check_equal(c2v(vec(mat)), [1, 2, 3, 4, 5, 6])


def tvec(mat):
    """transposed vec operator concatenates rows into single row matrix"""
    assert len(mat.shape) == 2
    return mat.reshape(1, -1)


def tvec_test():
    mat = torch.tensor([[1, 3, 5], [2, 4, 6]])
    check_equal(tvec(mat), [[1, 3, 5, 2, 4, 6]])


def unvec(a, rows):
    """reverse of vec, rows specifies number of rows in the final matrix."""
    assert len(a.shape) == 2
    assert a.shape[1] == 1
    assert a.shape[0] % rows == 0
    cols = a.shape[0] // rows
    return a.reshape(cols, -1).t()


def untvec(a, rows):
    """reverse of tvec, rows specifies number of rows in the final matrix."""
    assert len(a.shape) == 2
    assert a.shape[0] == 1
    assert a.shape[1] % rows == 0
    return a.reshape(rows, -1)


def kron(a, b):
    """Kronecker product."""
    return torch.einsum("ab,cd->acbd", a, b).view(a.size(0) * b.size(0), a.size(1) * b.size(1))


def slow_kron(a, b):
    """Slower version which is required when dimensions are not contiguous."""
    return torch.einsum("ab,cd->acbd", a, b).contiguous().view(a.size(0) * b.size(0), a.size(1) * b.size(1))


def kron_test():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[6, 7], [8, 9]])
    C = kron(A, B)
    Cnp = np.kron(to_numpy(A), to_numpy(B))
    check_equal(C, [[6, 7, 12, 14], [8, 9, 16, 18], [18, 21, 24, 28], [24, 27, 32, 36]])
    check_equal(C, Cnp)


def l2_norm(mat: torch.Tensor):
    u, s, v = torch.svd(mat)
    return torch.max(s)


def l2_norm_test():
    mat = torch.tensor([[1, 1], [0, 1]]).float()
    check_equal(l2_norm(mat), 0.5*(1+math.sqrt(5)))
    ii = torch.eye(5)
    check_equal(l2_norm(ii), 1)


def inv_square_root_numpy(mat):
    assert type(mat) == np.ndarray
    return scipy.linalg.inv(scipy.linalg.sqrtm(mat))


def pinv_square_root_numpy(mat):
    assert type(mat) == np.ndarray
    result = scipy.linalg.inv(scipy.linalg.sqrtm(mat))
    return result


def erank(mat):
    """Effective rank of matrix."""
    return torch.trace(mat) / l2_norm(mat)


def outer(x, y):
    return x.unsqueeze(1) @ y.unsqueeze(0)


def to_scalar(x):
    if hasattr(x, 'item'):
        return x.item()
    x = to_numpy(x).flatten()
    assert len(x) == 1
    return x[0]


def to_numpy(x, dtype=np.float32) -> np.ndarray:
    """Convert numeric object to numpy array."""
    if hasattr(x, 'numpy'):  # PyTorch tensor
        return x.detach().numpy().astype(dtype)
    elif type(x) == np.ndarray:
        return x.astype(dtype)
    else:  # Some Python type
        return np.array(x).astype(dtype)


def khatri_rao(A: torch.Tensor, B: torch.Tensor):
    """Khatri-Rao product.
     i'th column of result C_i is a Kronecker product of A_i and B_i

    Section 2.6 of Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM review 51.3
    (2009): 455-500"""
    assert A.shape[1] == B.shape[1]
    # noinspection PyTypeChecker
    return torch.einsum("ik,jk->ijk", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])


def test_khatri_rao():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[5, 12],  [7, 16],
                      [15, 24], [21, 32]])
    check_equal(khatri_rao(A, B), C)


def khatri_rao_t(A: torch.Tensor, B: torch.Tensor):
    """Transposed Khatri-Rao, inputs and outputs are transposed."""
    assert A.shape[0] == B.shape[0]
    # noinspection PyTypeChecker
    return torch.einsum("ki,kj->kij", A, B).reshape(A.shape[0], A.shape[1] * B.shape[1])


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
    check_equal(khatri_rao_t(A, B), C)


# Autograd functions, from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
# noinspection PyTypeChecker
def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y: torch.Tensor, x: torch.Tensor):
    return jacobian(jacobian(y, x, create_graph=True), x)


def pinv(mat: torch.Tensor, cond=None) -> torch.Tensor:
    """Computes pseudo-inverse of mat, treating eigenvalues below eps as 0.

        cond : float or None
        Cutoff for 'small' singular values. If omitted, singular values smaller
        than ``max(M,N)*largest_singular_value*eps`` are considered zero where
        ``eps`` is the machine precision.
        """

    # Take cut-off logic from scipy
    # https://github.com/ilayn/scipy/blob/0f4c793601ecdd74fc9826ac02c9b953de99403a/scipy/linalg/basic.py#L1307
    u, s, v = torch.svd(mat)
    if cond in [None, -1]:
        cond = torch.max(s) * max(mat.shape) * np.finfo(np.dtype('float32')).eps
    rank = torch.sum(s > cond)

    u = u[:, :rank]
    u /= s[:rank]
    return u @ v.t()[:rank]


def pinv_square_root(mat: torch.Tensor, eps=1e-4) -> torch.Tensor:
    u, s, v = torch.svd(mat)
    one = torch.from_numpy(np.array(1))
    ivals: torch.Tensor = one / torch.sqrt(s)
    si = torch.where(s > eps, ivals, s)
    return u @ torch.diag(si) @ v.t()


def check_close(a0, b0):
    return check_equal(a0, b0, rtol=1e-5, atol=1e-9)


def check_equal(observed, truth, rtol=1e-9, atol=1e-12):
    truth = to_numpy(truth)
    observed = to_numpy(observed)
    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol)

    # try:
    #     np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol)
    # except Exception as _e:
    #     print("Error" + "-" * 60)
    #     for line in traceback.format_stack():
    #         print(line.strip())
    #
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     print("*** print_tb:")
    #     traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
    #     efmt = traceback.format_exc()
    #     print(efmt)
    #     import pdb
    #     pdb.set_trace()


def get_param(layer):
    """Extract parameter out of layer, assumes there's just one parameter in a layer."""
    named_params = [(name, param) for (name, param) in layer.named_parameters()]
    assert len(named_params) == 1, named_params
    return named_params[0][1]


global_timeit_dict = {}


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
    it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        print(f"{interval_ms:8.2f}   {self.tag}")


def run_all_tests(module: nn.Module):
    all_functions = inspect.getmembers(module, inspect.isfunction)
    for name, func in all_functions:
        if name.endswith("_test"):
            with timeit(name):
                func()
    print(module.__name__+" tests passed.")


def freeze(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
    setattr(layer, "frozen", True)


def unfreeze(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True
    setattr(layer, "frozen", False)


if __name__ == '__main__':
    run_all_tests(sys.modules[__name__])


def nest_stats(tag: str, stats) -> Dict:
    """Nest given dict of stats under tag using TensorBoard syntax /nest1/tag"""
    result = {}
    for key, value in stats.items():
        result[f"{tag}/{key}"] = value
    return result


def seed_random(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zero_grad(model: nn.Module) -> None:
    """model.zero_grad + delete all backprops/activations/saved_grad values"""
    model.zero_grad()
    for m in model.modules():
        if hasattr(m, 'backprops'):
            del m.backprops
        if hasattr(m, 'activations'):
            del m.activations
    for p in model.parameters():
        if hasattr(p, 'saved_grad'):
            del p.saved_grad


class TinyMNIST(datasets.MNIST):
    """Custom-size MNIST autoencoder dataset for debugging."""

    def __init__(self, root, data_width=4, targets_width=4, dataset_size=60000, download=True):
        """

        Args:
            root: dataset root
            data_width: dimension of input images
            targets_width: dimension of target images
            dataset_size: number of examples
        """
        super().__init__(root, download)

        self.data = self.data[:dataset_size, :, :]
        new_data = np.zeros((self.data.shape[0], data_width, data_width))
        new_targets = np.zeros((self.data.shape[0], targets_width, targets_width))
        for i in range(self.data.shape[0]):
            arr = self.data[i, :].numpy().astype(np.uint8)
            im = Image.fromarray(arr)
            im.thumbnail((data_width, data_width), Image.ANTIALIAS)
            new_data[i, :, :] = np.array(im) / 255
            im = Image.fromarray(arr)
            im.thumbnail((targets_width, targets_width), Image.ANTIALIAS)
            new_targets[i, :, :] = np.array(im) / 255

        self.data = torch.from_numpy(new_data).float()
        self.data = self.data.unsqueeze(1)
        self.targets = torch.from_numpy(new_targets).float()
        self.targets = self.targets.unsqueeze(1)
        self.data, self.targets = self.data.to(gl.device), self.targets.to(gl.device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class SimpleNet(nn.Module):
    """Simple feedforward network that works on images."""
    def __init__(self, d: List[int], nonlin=False):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=False)
            setattr(linear, 'name', f'{i:02d}-linear')
            self.layers.append(linear)
            self.all_layers.append(linear)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)

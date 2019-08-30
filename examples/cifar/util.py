# Take simple example, plot per-layer stats over time
import inspect
import math
import os
import random
import sys
import time
from typing import Any, Dict, Callable, Optional
from typing import List

import globals as gl
import numpy as np
import scipy
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from PIL import Image

import torch.nn.functional as F

# to enable referring to functions in its own module as u.func
u = sys.modules[__name__]


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


def nan_check(mat):
    nan_mask = torch.isnan(mat).float()
    nans = torch.sum(nan_mask).item()
    not_nans = torch.sum(torch.tensor(1) - nan_mask).item()

    assert nans == 0, f"matrix of shape {mat.shape} has {nans}/{nans + not_nans} nans"


def has_nan(mat):
    return torch.sum(torch.isnan(mat)) > 0


def l2_norm(mat: torch.Tensor):
    u, s, v = torch.svd(mat)
    return torch.max(s)


def l2_norm_test():
    mat = torch.tensor([[1, 1], [0, 1]]).float()
    check_equal(l2_norm(mat), 0.5 * (1 + math.sqrt(5)))
    ii = torch.eye(5)
    check_equal(l2_norm(ii), 1)


def sym_l2_norm(mat: torch.Tensor):
    evals, _evecs = torch.symeig(mat)
    return torch.max(evals)


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


def sym_erank(mat):
    """Effective rank of matrix."""
    return torch.trace(mat) / sym_l2_norm(mat)


def lyapunov_svd(A, C, rtol=1e-4, use_svd=False):
    """Solve AX+XA=C"""

    assert A.shape[0] == A.shape[1]
    assert len(A.shape) == 2
    if use_svd:
        U, S, V = torch.svd(A)
    else:
        S, U = torch.symeig(A, eigenvectors=True)
    S = S.diag() @ torch.ones_like(A)
    X = U @ ((U.t() @ C @ U) / (S + S.t())) @ U.t()
    error = A @ X + X @ A - C
    relative_error = torch.max(torch.abs(error)) / torch.max(torch.abs(A))
    if relative_error > rtol:
        # TODO(y): currently spams with errors, implement another method based on Newton iteration
        pass
        # print(f"Warning, error {relative_error} encountered in lyapunov_svd")

    return X


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
    C = torch.tensor([[5, 12], [7, 16],
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

    nan_check(mat)
    u, s, v = torch.svd(mat)
    if cond in [None, -1]:
        cond = torch.max(s) * max(mat.shape) * np.finfo(np.dtype('float32')).eps
    rank = torch.sum(s > cond)

    u = u[:, :rank]
    u /= s[:rank]
    return u @ v.t()[:rank]


def pinv_square_root(mat: torch.Tensor, eps=1e-4) -> torch.Tensor:
    nan_check(mat)
    u, s, v = torch.svd(mat)
    one = torch.from_numpy(np.array(1))
    ivals: torch.Tensor = one / torch.sqrt(s)
    si = torch.where(s > eps, ivals, s)
    return u @ torch.diag(si) @ v.t()


def symsqrt(a, cond=None, return_rank=False):
    """Computes the symmetric square root of a positive definite matrix"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[a.dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B


def check_close(a0, b0, rtol=1e-5, atol=1e-8):
    return check_equal(a0, b0, rtol=rtol, atol=atol)


# todo: replace with torch.allclose
def check_equal(observed, truth, rtol=1e-9, atol=1e-12):
    truth = to_numpy(truth)
    observed = to_numpy(observed)
    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    assert np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True)
    #    np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol)

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
        # print(f"{interval_ms:8.2f}   {self.tag}")
        log_scalars({'time/' + self.tag: interval_ms})


def run_all_tests(module: nn.Module):
    class local_timeit:
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

    all_functions = inspect.getmembers(module, inspect.isfunction)
    for name, func in all_functions:
        if name.endswith("_test"):
            with local_timeit(name):
                func()
    print(module.__name__ + " tests passed.")


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


def clear_backprops(model: nn.Module) -> None:
    """model.zero_grad + delete all backprops/activations/saved_grad values"""
    model.zero_grad()
    for m in model.modules():
        if hasattr(m, 'backprops_list'):
            del m.backprops_list
        if hasattr(m, 'activations'):
            del m.activations
    for p in model.parameters():
        if hasattr(p, 'saved_grad'):
            del p.saved_grad


class TinyMNIST(datasets.MNIST):
    """Custom-size MNIST autoencoder dataset for debugging. Generates data/target images with reduced resolution.
    Use original_targets to get original MNIST labels"""

    def __init__(self, root, data_width=4, targets_width=4, dataset_size=0, download=True, train=True,
                 original_targets=False):
        """

        Args:
            root: dataset root
            data_width: dimension of input images
            targets_width: dimension of target images
            dataset_size: number of examples
            original_targets: if False, replaces original classification targets with image reconstruction targets
        """
        super().__init__(root, download=download, train=train)

        if dataset_size > 0:
            assert dataset_size <= self.data.shape[0]
            self.data = self.data[:dataset_size, :, :]

        if data_width != 28 or targets_width != 28:
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
            if not original_targets:
                self.targets = torch.from_numpy(new_targets).float()
        else:
            self.data = self.data.float().unsqueeze(1)
            self.targets = self.data
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


model_layer_map = {}
model_param_map = {}


class SimpleModel(nn.Module):
    """Simple sequential model. Adds layers[] attribute, flags to turn on/off hooks, and lookup mechanism from layer to parent
    model."""

    layers: List[nn.Module]
    all_layers: List[nn.Module]
    skip_forward_hooks: bool
    skip_backward_hooks: bool

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.skip_backward_hooks = False
        self.skip_forward_hooks = False

    def _finalize(self):
        global model_layer_map
        for module in self.modules():
            model_layer_map[module] = self
        for param in self.parameters():
            model_layer_map[param] = self


def get_parent_model(module_or_param) -> Optional[nn.Module]:
    """Returns root model for given parameter."""
    global model_layer_map
    global model_param_map
    if module_or_param in model_layer_map:
        assert module_or_param not in model_param_map
        return model_layer_map[module_or_param]
    if module_or_param in model_param_map:
        return model_param_map[module_or_param]


class SimpleFullyConnected(SimpleModel):
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

        super()._finalize()

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)


class SimpleConvolutional(SimpleModel):
    """Simple conv network."""

    def __init__(self, d: List[int], kernel_size=(2, 2), nonlin=False):
        """

        Args:
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            conv = nn.Conv2d(d[i], d[i + 1], kernel_size, bias=False)
            setattr(conv, 'name', f'{i:02d}-conv')
            self.layers.append(conv)
            self.all_layers.append(conv)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

        self._finalize()

    def forward(self, x: torch.Tensor):
        return self.predict(x)


def log_scalars(metrics: Dict[str, Any]) -> None:
    for tag in metrics:
        gl.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=gl.token_count)


def log_scalar(**metrics) -> None:
    for tag in metrics:
        gl.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=gl.token_count)


def get_events(fname, x_axis='step'):
    """Returns event dictionary for given run, has form
    {tag1: {step1: val1}, tag2: ..}

    If x_axis is set to "time", step is replaced by timestamp
    """

    from tensorflow.python.summary import summary_iterator  # local import because TF is heavy dep and only used here

    result = {}

    events = summary_iterator.summary_iterator(fname)

    try:
        for event in events:
            if x_axis == 'step':
                x_val = event.step
            elif x_axis == 'time':
                x_val = event.wall_time
            else:
                assert False, f"Unknown x_axis ({x_axis})"

            vals = {val.tag: val.simple_value for val in event.summary.value}
            # step_time: value
            for tag in vals:
                event_dict = result.setdefault(tag, {})
                if x_val in event_dict:
                    print(f"Warning, overwriting {tag} for {x_axis}={x_val}")
                    print(f"old val={event_dict[x_val]}")
                    print(f"new val={vals[tag]}")

                event_dict[x_val] = vals[tag]
    except Exception as e:
        print(e)
        pass

    return result


def infinite_iter(obj):
    while True:
        for result in iter(obj):
            yield result


# noinspection PyTypeChecker
def dump(result, fname):
    """Save result to file. Load as np.genfromtxt(fname). """
    result = to_numpy(result)
    if result.shape == ():  # savetxt has problems with scalars
        result = np.expand_dims(result, 0)
    location = fname
    # special handling for integer datatypes
    if (
            result.dtype == np.uint8 or result.dtype == np.int8 or
            result.dtype == np.uint16 or result.dtype == np.int16 or
            result.dtype == np.uint32 or result.dtype == np.int32 or
            result.dtype == np.uint64 or result.dtype == np.int64
    ):
        np.savetxt(location, X=result, fmt="%d", delimiter=',')
    else:
        np.savetxt(location, X=result, delimiter=',')
    print("Dumping to", location)


def print_version_info():
    """Print version numbers of numerical packages in current env."""

    def get_mkl_version():
        import ctypes
        import numpy as np

        # this recipe only works on Linux
        try:
            ver = np.zeros(199, dtype=np.uint8)
            mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
            mkl.MKL_Get_Version_String(ver.ctypes.data_as(ctypes.c_char_p), 198)
            return ver[ver != 0].tostring()
        except:
            return 'unknown'

    if np.__config__.get_info("lapack_mkl_info"):
        print("MKL version", get_mkl_version())
    else:
        print("not using MKL")

    print("PyTorch version", torch.version.__version__)

    print("Scipy version: ", scipy.version.full_version)
    print("Numpy version: ", np.version.full_version)
    print_cpu_info()


def print_cpu_info():
    ver = 'unknown'
    try:
        for l in open("/proc/cpuinfo").read().split('\n'):
            if 'model name' in l:
                ver = l
                break
    except:
        pass

    # core counts from https://stackoverflow.com/a/23378780/419116
    print("CPU version: ", ver)
    sys.stdout.write("CPU logical cores: ")
    sys.stdout.flush()
    os.system(
        "echo $([ $(uname) = 'Darwin' ] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)")
    sys.stdout.write("CPU physical cores: ")
    sys.stdout.flush()
    os.system(
        "echo $([ $(uname) = 'Darwin' ] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)")

    # get mapping of logical cores to physical sockets
    import re
    socket_re = re.compile(
        """.*?processor.*?(?P<cpu>\\d+).*?physical id.*?(?P<socket>\\d+).*?power""",
        flags=re.S)
    from collections import defaultdict
    socket_dict = defaultdict(list)
    try:
        for cpu, socket in socket_re.findall(open('/proc/cpuinfo').read()):
            socket_dict[socket].append(cpu)
    except FileNotFoundError:
        pass
    print("CPU physical sockets: ", len(socket_dict))


def move_to_gpu(tensors):
    return [tensor.cuda() for tensor in tensors]


# Functions to capture backprops/activations and save them on the layer

# layer.register_forward_hook(capture_activations) -> saves activations/output as layer.activations/layer.output
# layer.register_backward_hook(capture_backprops)  -> appends each backprop to layer.backprops_list
# layer.weight.register_hook(save_grad(layer.weight)) -> saves grad under layer.weight.saved_grad
# util.clear_backprops(model) -> delete all values above

def capture_activations(module: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Saves activations (layer input) into layer.activations. """
    model = get_parent_model(module)
    if getattr(model, 'skip_forward_hooks', False):
        return
    assert not hasattr(module,
                       'activations'), "Seeing results of previous autograd, call util.clear_backprops(model) to clear"
    assert len(input) == 1, "this was tested for single input layers only"
    setattr(module, "activations", input[0].detach())
    setattr(module, "output", output.detach())


def capture_backprops(module: nn.Module, _input, output):
    """Appends all backprops (Jacobian Lops from upstream) to layer.backprops_list.
    Using list in order to capture multiple backprop values for a single batch. Use util.clear_backprops(model)
    to clear all saved values.
    """
    model = get_parent_model(module)
    if getattr(model, 'skip_backward_hooks', False):
        return
    assert len(output) == 1, "this works for single variable layers only"
    if not hasattr(module, 'backprops_list'):
        setattr(module, 'backprops_list', [])
    assert len(module.backprops_list) < 100, "Possible memory leak, captured more than 100 backprops, comment this assert " \
                                             "out if this is intended."""

    module.backprops_list.append(output[0])


def save_grad(param: nn.Parameter) -> Callable[[torch.Tensor], None]:
    """Hook to save gradient into 'param.saved_grad', so it can be accessed after model.zero_grad(). Only stores gradient
    if the value has not been set, call util.clear_backprops to clear it."""

    def save_grad_fn(grad):
        if not hasattr(param, 'saved_grad'):
            setattr(param, 'saved_grad', grad)

    return save_grad_fn


def fmt(a):
    """Helper function for converting copy-pasted Mathematica matrices into Python."""

    a = a.replace('\n', '')
    print(a.replace("{", "[").replace("}", "]"))


def to_logits(p: torch.Tensor) -> torch.Tensor:
    """Inverse of F.softmax"""
    if len(p.shape) == 1:
        batch = torch.unsqueeze(p, 0)
    else:
        assert len(p.shape) == 2
        batch = p

    batch = torch.log(batch) - torch.log(batch[:, -1])
    return batch.reshape(p.shape)


class CrossEntropySoft(nn.Module):
    """Like torch.nn.CrossEntropyLoss but instead of class index it accepts a
    probability distribution.

    The `input` is expected to contain raw, unnormalized scores for each class.
    The `target` is expected to contain empirical probabilities for each class (positive and adding up to 1)

    """

    def __init__(self):
        super(CrossEntropySoft, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """

        # check that targets are positive and add up to 1
        assert (target >= 0).sum() == target.numel()
        sums = target.sum(dim=1)
        assert np.allclose(sums, torch.ones_like(sums))

        assert len(target.shape) == 2
        n = target.shape[0]
        log_likelihood = -F.log_softmax(inputs, dim=1)
        loss = torch.sum(torch.mul(target, log_likelihood)) / n

        return loss


def get_unique_logdir(root_logdir: str) -> str:
    """Increments suffix at the end of root_logdir until getting directory that doesn't exist locally, return that."""
    count = 0
    while os.path.exists(f"{root_logdir}{count:02d}"):
        count += 1
    return f"{root_logdir}{count:02d}"


######################################################
# Hessian backward samplers
#
# A sampler provides a representation of hessian of the loss layer
#
# For a batch of size n,o, Hessian backward sampler will produce k backward values
# (k between 1 and o) where each value can be fed into model.backward(value)
#
# The gradients corresponding to these k backward will be summed up to form an estimate of Hessian of the network
# sum_i gg' \approx H
#
#   sampler = HessianSamplerMyLoss
#   for bval in sampler(model(batch)):
#       model.zero_grad()
#       model.backward(bval)

class HessianSampler:
    pass


class HessianExactSqrLoss(HessianSampler):
    """Sampler for loss err*err/2/len(batch), produces exact Hessian."""

    def __init__(self):
        super().__init__()

    def __call__(self, output: torch.Tensor):
        assert len(output.shape) == 2
        batch_size, output_size = output.shape
        id_mat = torch.eye(output_size)
        for out_idx in range(output_size):
            yield torch.stack([id_mat[out_idx]] * batch_size)


class HessianSampledSqrLoss(HessianSampler):
    """Sampler for loss err*err/2/len(batch), produces exact Hessian."""

    samples: int

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __call__(self, output: torch.Tensor):
        assert len(output.shape) == 2
        batch_size, output_size = output.shape
        assert self.samples <= output_size, f"Requesting more samples than needed for exact Hessian computation " \
            f"({self.samples}>{output_size})"

        # exact sampler provides n samples whose outer products add up to Identity
        # this sampler can provides a single sample where outer product is Identity in expectation
        # therefore must divide each individual vector in outer product by sqrt(samples)

        for out_idx in range(self.samples):
            # sample random vectors of +1/-1's
            bval = torch.LongTensor(batch_size, output_size).to(gl.device).random_(0, 2) * 2 - 1
            yield bval.float()/math.sqrt(self.samples)


class HessianExactCrossEntropyLoss(HessianSampler):
    """Sampler for nn.CrossEntropyLoss, produces exact Hessian."""

    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor):
        assert len(logits.shape) == 2

        n, d = logits.shape
        batch = F.softmax(logits, dim=1)

        mask = torch.eye(d).expand(n, d, d)
        diag_part = batch.unsqueeze(2).expand(n, d, d) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, d, d)

        for i in range(n):
            hess[i, :, :] = u.symsqrt(hess[i, :, :])

        for out_idx in range(d):
            sample = hess[:, out_idx, :]
            assert sample.shape == (n, d)
            yield sample


def register_hooks(model: SimpleModel):
    # TODO(y): remove hardcoding of parameter name
    for layer in model.layers:
        layer.register_forward_hook(u.capture_activations)
        layer.register_backward_hook(u.capture_backprops)
        layer.weight.register_hook(u.save_grad(layer.weight))


def hessian_from_backprops(A_t, Bh_t):
    """Computes Hessian from a batch of forward and backward values.

    See documentation on HessianSampler for assumptions on how backprop values are generated

    For batch size n
    Forward values have shape n,layer_inputs
    Backward values is a list of length c of tensors of shape n,layer_outputs

    For exact Hessian computation, c is number of classes
    """
    n = A_t.shape[0]
    Amat_t = torch.cat([A_t] * len(Bh_t), dim=0)  # todo: can instead replace with a khatri-rao loop
    Bmat_t = torch.cat(Bh_t, dim=0)
    Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch Jacobian
    H = Jb.t() @ Jb / n
    return H


def kl_div_cov(mat1, mat2, eps=1e-3):
    """KL divergence between two zero centered Gaussian's with given covariance matrices."""

    evals1 = torch.symeig(mat1).eigenvalues
    evals2 = torch.symeig(mat2).eigenvalues
    k = mat1.shape[0]
    # scale regularizer in proportion to achievable numerical precision (taken from scipy.pinv2)
    l1 = torch.max(evals1) * k
    l2 = torch.max(evals2) * k
    l = max(l1, l2)
    reg = torch.eye(mat1.shape[0])*l*eps
    mat1 = mat1 + reg
    mat2 = mat2 + reg

    div = torch.trace(mat1@torch.inverse(mat2))-(torch.logdet(mat1)-torch.logdet(mat2))-k
    return div

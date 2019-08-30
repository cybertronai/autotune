import os
import sys
import time
from typing import Optional, Tuple, Callable

# import torch
import scipy
import torch
from torchcurv.optim import SecondOrderOptimizer

import torch.nn as nn

module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path)
import util as u

import numpy as np

class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result


def lyapunov_test():
    """Test that scipy lyapunov solver works correctly."""
    d = 2
    n = 3
    model = Net(d)

    w0 = torch.tensor([[1, 2]]).float()
    assert w0.shape[1] == d
    model.w.weight.data.copy_(w0)

    X = torch.tensor([[-2, 0, 2], [-1, 1, 3]]).float()
    assert X.shape[0] == d
    assert X.shape[1] == n

    Y = torch.tensor([[0, 1, 2]]).float()
    assert Y.shape[1] == X.shape[1]

    data = X.t()  # PyTorch expects batch dimension first
    target = Y.t()
    assert data.shape[0] == n

    output = model(data)
    # residuals, aka e
    residuals = output - Y.t()

    def compute_loss(residuals_):
        return torch.sum(residuals_ * residuals_) / (2 * n)

    loss = compute_loss(residuals)

    assert loss - 8.83333 < 1e-5, torch.norm(loss) - 8.83333

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0, momentum=0, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", )
    curv_args = dict(damping=1, ema_decay=1)  # todo: damping
    optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)

    def backward(last_layer: str) -> Callable:
        """Creates closure that backpropagates either from output layer or from loss layer"""

        def closure() -> Tuple[Optional[torch.Tensor], torch.Tensor]:
            optimizer.zero_grad()
            output = model(data)
            if last_layer == "output":
                output.backward(torch.ones_like(target))
                return None, output
            elif last_layer == 'loss':
                loss = compute_loss(output - target)
                loss.backward()
                return loss, output
            else:
                assert False, 'last layer must be "output" or "loss"'

        return closure

    #    loss = compute_loss(output - Y.t())
    #    loss.backward()

    loss, output = optimizer.step(closure=backward('loss'))
    J = X.t()
    A = model.w.data_input
    B = model.w.grad_output * n
    G = residuals.repeat(1, d) * J
    losses = torch.stack([compute_loss(r) for r in residuals])
    g = G.sum(dim=0) / n
    efisher = G.t() @ G / n
    sigma = efisher - u.outer(g, g)
    loss2 = (residuals * residuals).sum() / (2 * n)
    H = J.t() @ J / n
    noise_variance = torch.trace(H.inverse() @ sigma)


    # Slow way
    p_sigma = (u.kron(H, torch.eye(d)) + u.kron(torch.eye(d), H)).inverse() @ u.vec(sigma)
    p_sigma = u.unvec(p_sigma, d)

    sigma0 = u.to_numpy(sigma)
    H0 = u.to_numpy(H)


    # Alternative fast way
    p_sigma2 = scipy.linalg.solve_lyapunov(H0, sigma0)
    u.check_close(p_sigma, p_sigma2)


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
        print(f"{interval_ms:8.2f}   {self.tag}")


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


def print_cpu_info():
  ver = 'unknown'
  try:
    for l in open("/proc/cpuinfo").read().split('\n'):
      if 'model name' in l:
        ver = l
        break
  except:
    pass


def linalg_bench():
    if np.__config__.get_info("lapack_mkl_info"):
        print("MKL version", get_mkl_version())
    else:
        print("not using MKL")

    print("PyTorch version", torch.version.__version__)

    print("Scipy version: ", scipy.version.full_version)
    print("Numpy version: ", np.version.full_version)

    for d in [1024]:
        print(f"{d}-by-{d} matrix")
        n = 10000
        assert n > 2*d   # to prevent singularity
        X = np.random.random((d, 10000))
        Y = np.random.random((d, 10000))
        H = X @ X.T
        S = Y @ Y.T

        with timeit(f"linalg.solve_lyapunov"):
            result = scipy.linalg.solve_lyapunov(H, S)
            #print(result[0,0])

        with timeit(f"linalg.pinvh"):
            result = scipy.linalg.pinvh(H)
            #print(result[0, 0])

        with timeit(f"linalg.pinv"):
            result = scipy.linalg.pinv(H)
            #print(result[0, 0])


        with timeit(f"linalg.inv"):
            result = scipy.linalg.inv(H)
            #print(result[0, 0])



if __name__ == '__main__':
    linalg_bench()

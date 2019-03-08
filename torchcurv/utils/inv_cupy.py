import numpy

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import util

from torch.utils.dlpack import to_dlpack, from_dlpack

import warnings

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver

use_cholesky = True

# Based cupy (cupy/cupy/linalg/solve.py) @ 067f830


def inv(m):
    m_cp = cupy.fromDlpack(to_dlpack(m))
    m_inv_cp = inv_core(m_cp, use_cholesky)
    return from_dlpack(m_inv_cp.toDlpack())


def inv_core(a, cholesky=False):
    """Computes the inverse of a matrix.
    This function computes matrix ``a_inv`` from n-dimensional regular matrix
    ``a`` such that ``dot(a, a_inv) == eye(n)``.
    Args:
        a (cupy.ndarray): The regular matrix
        b (Boolean): Use cholesky decomposition
    Returns:
        cupy.ndarray: The inverse of a matrix.
    .. seealso:: :func:`numpy.linalg.inv`
    """

    xp = cupy.get_array_module(a)
    if xp == numpy:
        if cholesky:
            warnings.warn(
                "Current fast-inv using cholesky doesn't support numpy.ndarray.")
        return numpy.linalg.inv(a)

    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # to prevent `a` to be overwritten
    a = a.copy()

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=cupy.int)
    m = a.shape[0]

    b = cupy.eye(m, dtype=dtype)

    if not cholesky:
        if dtype == 'f':
            getrf = cusolver.sgetrf
            getrf_bufferSize = cusolver.sgetrf_bufferSize
            getrs = cusolver.sgetrs
        else:  # dtype == 'd'
            getrf = cusolver.dgetrf
            getrf_bufferSize = cusolver.dgetrf_bufferSize
            getrs = cusolver.dgetrs

        buffersize = getrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)

        # TODO(y1r): cache buffer to avoid malloc
        workspace = cupy.empty(buffersize, dtype=dtype)
        ipiv = cupy.empty((a.shape[0], 1), dtype=dtype)

        # LU Decomposition
        getrf(cusolver_handle, m, m, a.data.ptr, m,
              workspace.data.ptr, ipiv.data.ptr, dev_info.data.ptr)

        # TODO(y1r): check dev_info status

        # solve for the inverse
        getrs(cusolver_handle, 0, m, m, a.data.ptr, m,
              ipiv.data.ptr, b.data.ptr, m, dev_info.data.ptr)

        # TODO(y1r): check dev_info status
    else:
        if dtype == 'f':
            potrf = cusolver.spotrf
            potrf_bufferSize = cusolver.spotrf_bufferSize
            potrs = cusolver.spotrs
        else:  # dtype == 'd'
            potrf = cusolver.dpotrf
            potrf_bufferSize = cusolver.dpotrf_bufferSize
            potrs = cusolver.dpotrs

        buffersize = potrf_bufferSize(
            cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, m, a.data.ptr, m)

        # TODO(y1r): cache buffer to avoid malloc
        workspace = cupy.empty(buffersize, dtype=dtype)

        # Cholesky Decomposition
        potrf(cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, m,
              a.data.ptr, m, workspace.data.ptr, buffersize, dev_info.data.ptr)

        # TODO(y1r): check dev_info status

        # solve for the inverse
        potrs(cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, m,
              m, a.data.ptr, m, b.data.ptr, m, dev_info.data.ptr)

        # TODO(y1r): check dev_info status

    return b

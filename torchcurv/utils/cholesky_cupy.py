import cupy

from torchcurv.utils.cupy import to_cupy, from_cupy


def cholesky(m, upper=True):
    m_cp = to_cupy(m)
    m_chl_cp = cupy.linalg.decomposition.cholesky(m_cp)
    if upper:
        m_chl_cp = m_chl_cp.transpose()
    return from_cupy(m_chl_cp)

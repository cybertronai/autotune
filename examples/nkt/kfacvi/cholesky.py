
import cupy

from torch.utils.dlpack import to_dlpack,from_dlpack


def cholesky(m,upper=True):
    m_cp = cupy.fromDlpack(to_dlpack(m))
    chl_cp = cupy.linalg.decomposition.cholesky(m_cp)
    if upper:
        chl_cp = chl_cp.transpose()
    return from_dlpack(chl_cp.toDlpack())

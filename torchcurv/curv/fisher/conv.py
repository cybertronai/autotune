import torch
from torchcurv import KronCovConv2d, Fisher


class KronFisherConv2d(KronCovConv2d, Fisher):

    def __init__(self, *args, **kwargs):
        KronCovConv2d.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_output):
        if self.do_backward:
            assert self.prob is not None
            n, c, h, w = grad_output.shape  # n x c_out x h_out x w_out

            pg = torch.mul(grad_output, self.prob.reshape(n, 1, 1, 1))
            pm = pg.transpose(0, 1).reshape(c, -1)  # c_out x n(h_out)(w_out)
            m = grad_output.transpose(0, 1).reshape(c, -1)  # c_out x n(h_out)(w_out)

            G = torch.einsum('ik,jk->ij', pm, m).div(n*h*w)  # c_out x c_out
            self._G = G
            self.accumulate_cov(G)
        else:
            self._G = self.finalize()

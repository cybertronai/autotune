import torch
from torchcurv import KronCovLinear, Fisher


class KronFisherLinear(KronCovLinear, Fisher):

    def __init__(self, *args, **kwargs):
        KronCovLinear.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_output):
        if self.do_backward:
            assert self.prob is not None
            n = grad_output.shape[0]  # n x f_out

            pg = torch.mul(grad_output, self.prob.reshape(n, 1))

            # f_out x f_out
            G = torch.einsum(
                'ki,kj->ij', pg, grad_output).div(n)
            self._G = G
            self.accumulate_cov(G)
        else:
            self._G = self.finalize()

from torchcurv.curv import Curvature, DiagCurvature, KronCurvature

import torch

from .inv import inv


class FisherLinear(Curvature):

    def __init__(self):
        pass


class DiagFisherLinear(DiagCurvature):

    def __init__(self):
        pass


class KronFisherLinear(KronCurvature):

    def __init__(self):
        pass

    def compute_G(self):
        grad_output_data = self._module.grad_output_data
        batch_size = grad_output_data.shape[0]
        return grad_output_data.transpose(0, 1).mm(grad_output_data).mul(1/batch_size)

    def compute_A(self):
        input_data = self._module.input_data
        batch_size = input_data.shape[0]
        if self.bias:
            input_data = torch.cat((input_data, torch.ones(
                (batch_size, 1), device=input_data.device)), 1)
        return input_data.transpose(0, 1).mm(input_data).mul(1/batch_size)

    def compute_precgrad(self, params):
        # update covs
        A, G = self.compute_A(), self.compute_G()
        self.covs = A, G
        # update covs_ema
        if self.cov_ema_decay != 0:
            self.update_covs_ema()
            A, G = self.covs_ema

        A, G = self.compute_damped_covs((A, G))

        A_inv, G_inv = inv(A), inv(G)

        # todo check params == list?
        if self.bias:
            param_grad = torch.cat(
                (params[0].grad, params[1].grad.view(-1, 1)), 1)
            precgrad = G_inv.mm(param_grad).mm(A_inv)
            return precgrad[:, 0:-1], precgrad[:, -1]
        else:
            param_grad = params[0].grad
            precgrad = G_inv.mm(param_grad).mm(A_inv)
            return (precgrad,)

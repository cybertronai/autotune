from torchcurv import Curvature, DiagCurvature, KronCurvature

import torch


class FisherLinear(Curvature):

    def __init__(self):
        pass


class DiagFisherLinear(DiagCurvature):

    def update(self, input_data, grad_output_data):
        pass


class KronFisherLinear(KronCurvature):

    def update_A(self, input_data):
        batch_size = input_data.shape[0]
        if self.bias:
            input_data = torch.cat((input_data, torch.ones(
                (batch_size, 1), device=input_data.device)), 1)
        self._A = input_data.transpose(0, 1).mm(input_data).mul(1/batch_size)

    def update_G(self, grad_output_data):
        batch_size = grad_output_data.shape[0]
        scale = batch_size  # for adjusting grad scale along with 'reduction' in loss function

        self._G = grad_output_data.transpose(0, 1).mm(
            grad_output_data).mul(1/batch_size).mul(scale**2)

    def compute_precgrad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        if self.bias:
            param_grad = torch.cat(
                (params[0].grad, params[1].grad.view(-1, 1)), 1)
            precgrad = G_inv.mm(param_grad).mm(A_inv)
            return [precgrad[:, 0:-1], precgrad[:, -1]]
        else:
            param_grad = params[0].grad
            precgrad = G_inv.mm(param_grad).mm(A_inv)
            return [precgrad]

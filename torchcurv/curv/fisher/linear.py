import torch
from torchcurv import Curvature, DiagCurvature, KronCurvature


class FisherLinear(Curvature):

    def update_in_backward(self, grad_output_data):
        pass

    def precondition_grad(self, params):
        pass


class DiagFisherLinear(DiagCurvature):

    def update_in_backward(self, grad_output_data):
        input_data = self._input_data  # n x f_in
        n = input_data.shape[0]

        in_in = input_data.mul(input_data)  # n x f_in
        grad_grad = grad_output_data.mul(grad_output_data)  # n x f_out

        data_w = torch.einsum('ki,kj->ij', grad_grad,
                              in_in).div(n)  # f_out x f_in
        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # f_out x 1
            self._data.append(data_b)


class KronFisherLinear(KronCurvature):

    def update_in_forward(self, input_data):
        n = input_data.shape[0]  # n x f_in
        if self.bias:
            ones = input_data.new_ones((n, 1))
            # shape: n x (f_in+1)
            input_data = torch.cat((input_data, ones), 1)

        # f_in x f_in or (f_in+1) x (f_in+1)
        A = torch.einsum('ki,kj->ij', input_data, input_data).div(n)
        self._A = A

    def update_in_backward(self, grad_output_data):
        n = grad_output_data.shape[0]  # n x f_out

        # f_out x f_out
        G = torch.einsum(
            'ki,kj->ij', grad_output_data, grad_output_data).div(n)
        self._G = G

    def precondition_grad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        if self.bias:
            grad = torch.cat(
                (params[0].grad, params[1].grad.view(-1, 1)), 1)
            precgrad = G_inv.mm(grad).mm(A_inv)

            setattr(params[0], 'precgrad', precgrad[:, :-1])
            setattr(params[1], 'precgrad', precgrad[:, -1])
        else:
            grad = params[0].grad
            precgrad = G_inv.mm(grad).mm(A_inv)

            setattr(params[0], 'precgrad', precgrad)

    def sample_params(self, params, mean, std_scale):
        A_ic, G_ic = self.std

        if self.bias:
            m = torch.cat(
                (mean[0], mean[1].view(-1, 1)), 1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data.copy_(param[:, 0:-1])
            params[1].data.copy_(param[:, -1])
        else:
            m = mean[0]
            param = mean.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data = param

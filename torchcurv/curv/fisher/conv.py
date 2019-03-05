from torchcurv.curv import Curvature, DiagCurvature, KronCurvature
import torch
import torch.nn.functional as F


class FisherConv2d(Curvature):

    def __init__(self):
        pass


class DiagFisherConv2d(DiagCurvature):

    def __init__(self):
        pass


class KronFisherConv2d(KronCurvature):

    def update_A(self, input_data):
        kernel_size, stride, padding, dilation = \
            self._module.kernel_size, self._module.stride, self._module.padding, self._module.dilation
        input_data2d = F.unfold(input_data, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        batch_size, a, _ = input_data2d.shape
        m = input_data2d.transpose(0, 1).reshape(a, -1)
        a, b = m.shape
        if self.bias:
            m = torch.cat((m, torch.ones((1, b), device=input_data.device)), 0)
        self._A = m.mm(m.transpose(0, 1)).mul(1/batch_size)

    def update_G(self, grad_output_data):
        batch_size, c, h, w = grad_output_data.shape
        m = grad_output_data.transpose(0, 1).reshape(c, -1)

        self._G = m.mm(m.transpose(0, 1)).mul(1/(batch_size*h*w))

    def compute_precgrad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        oc, ic, h, w = params[0].shape
        if self.bias:
            param_grad2d = torch.cat(
                (params[0].grad.reshape(oc, -1), params[1].grad.view(-1, 1)), 1)
            precgrad2d = G_inv.mm(param_grad2d).mm(A_inv)
            return [precgrad2d[:, 0:-1].reshape(oc, ic, h, w), precgrad2d[:, -1]]
        else:
            param_grad2d = params[0].grad.reshape(oc, -1)
            precgrad2d = G_inv.mm(param_grad2d).mm(A_inv)
            return [precgrad2d.reshape(oc, ic, h, w)]

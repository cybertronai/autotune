from torchcurv.curv import Curvature, DiagCurvature, KronCurvature
import torch
import torch.nn.functional as F

from torchcurv.utils import inv

class FisherConv2d(Curvature):

    def __init__(self):
        pass


class DiagFisherConv2d(DiagCurvature):

    def __init__(self):
        pass


class KronFisherConv2d(KronCurvature):

    def __init__(self, kernel_size, stride, padding, dilation, damping=0.01, cov_ema_decay=0.99, bias=True, pi_type='trace_norm'):
        super(KronFisherConv2d, self).__init__(damping, cov_ema_decay, bias, pi_type)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def compute_G(self, grad_output_data):
        batch_size, c, h, w = grad_output_data.shape
        m = grad_output_data.transpose(0, 1).reshape(c, -1)

        return m.mm(m.transpose(0, 1)).mul(1/(batch_size*h*w))

    def compute_A(self, input_data):
        input_data2d = F.unfold(input_data, kernel_size=self.kernel_size,
                                stride=self.stride, padding=self.padding, dilation=self.dilation)
        batch_size, a, _ = input_data2d.shape
        m = input_data2d.transpose(0, 1).reshape(a, -1)
        a, b = m.shape
        if self.bias:
            m = torch.cat((m, torch.ones((1, b), device=input_data.device)), 0)

        return m.mm(m.transpose(0, 1)).mul(1/batch_size)

    def compute_precgrad(self, tp, kfac_buf):
        A, G = self.compute_A(), self.compute_G()
        if self.cov_ema_decay != 0:
            self.update_covs_ema((A, G))
            A, G = self.covs_ema

        A, G = self.compute_damped_covs((A, G))

        A_inv, G_inv = inv(A), inv(G)

        param_grad = tp[0].grad
        oc, ic, h, w = param_grad.shape
        param_grad2d = param_grad.reshape(oc, -1)
        if self.bias:
            param_grad2d = torch.cat((param_grad2d, tp[1].grad.view(-1, 1)), 1)
            precgrad2d = G_inv.mm(param_grad2d).mm(A_inv)
            return precgrad2d[:, 0:-1].reshape(oc, ic, h, w), precgrad2d[:, -1]
        else:
            precgrad2d = G_inv.mm(param_grad2d).mm(A_inv)
            return (precgrad2d.reshape(oc, ic, h, w),)

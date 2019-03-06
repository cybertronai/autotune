from torchcurv import Curvature, DiagCurvature, KronCurvature
import torch
import torch.nn.functional as F


class FisherConv2d(Curvature):

    def update_in_backward(self, grad_output_data):
        pass

    def precgrad(self, params):
        pass


class DiagFisherConv2d(DiagCurvature):

    def update_in_backward(self, grad_output_data):
        input_data = self._input_data  # n x c_in x h_in x w_in
        conv2d = self._module

        # n x (c_in)(k_h)(k_w) x (h_out)(w_out)
        input_data2d = F.unfold(input_data,
                                kernel_size=conv2d.kernel_size, stride=conv2d.stride,
                                padding=conv2d.padding, dilation=conv2d.dilation)
        n, ckk, hw = input_data2d.shape

        # (c_in)(k_h)(k_w) x n(h_out)(w_out)
        input_data2d = input_data2d.transpose(0, 1).reshape(ckk, -1)

        # n x c_out x h_out x w_out
        n, c_out, h, w = grad_output_data.shape
        # c_out x n(h_out)(w_out)
        grad_output_data2d = grad_output_data.transpose(0, 1).reshape(c_out, -1)

        in_in = input_data2d.mul(input_data2d)  # (c_in)(k_h)(k_w) x n(h_out)(w_out)
        grad_grad = grad_output_data2d.mul(grad_output_data2d)  # c_out x n(h_out)(w_out)

        data_w = torch.einsum('ik,jk->ij', grad_grad, in_in).div(n*h*w)  # c_out x (c_in)(k_h)(k_w)
        data_w = data_w.reshape((c_out, -1, *conv2d.kernel_size))  # c_out x c_in x k_h x k_w
        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=1)  # c_out x 1
            self._data.append(data_b)


class KronFisherConv2d(KronCurvature):

    def update_in_forward(self, input_data):
        kernel_size, stride, padding, dilation = \
            self._module.kernel_size, self._module.stride, self._module.padding, self._module.dilation
        input_data2d = F.unfold(input_data, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        n, a, _ = input_data2d.shape
        m = input_data2d.transpose(0, 1).reshape(a, -1)
        a, b = m.shape
        if self.bias:
            m = torch.cat((m, m.new_ones((1, b))), 0)

        self._A = torch.einsum('ik,jk->ij', m, m).div(n)

    def update_in_backward(self, grad_output_data):
        n, c, h, w = grad_output_data.shape  # n x c x h x w
        m = grad_output_data.transpose(0, 1).reshape(c, -1)  # c x nhw

        self._G = torch.einsum('ik,jk->ij', m, m).div(n*h*w)

    def precgrad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        oc, ic, h, w = params[0].shape
        if self.bias:
            grad2d = torch.cat(
                (params[0].grad.reshape(oc, -1), params[1].grad.view(-1, 1)), 1)
            precgrad2d = G_inv.mm(grad2d).mm(A_inv)

            return [precgrad2d[:, 0:-1].reshape(oc, ic, h, w), precgrad2d[:, -1]]
        else:
            grad2d = params[0].grad.reshape(oc, -1)
            precgrad2d = G_inv.mm(grad2d).mm(A_inv)

            return [precgrad2d.reshape(oc, ic, h, w)]

from torchcurv import Curvature, DiagCurvature


class FisherBatchNorm1d(Curvature):

    def update_in_backward(self, grad_output_data):
        pass


class DiagFisherBatchNorm1d(DiagCurvature):

    def update_in_backward(self, grad_output_data):
        input_data = self._input_data  # n x f

        in_in = input_data.mul(input_data)  # n x f
        grad_grad = grad_output_data.mul(grad_output_data)  # n x f

        data_w = in_in.mul(grad_grad).mean(dim=0)  # f x 1

        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # f x 1
            self._data.append(data_b)


class FisherBatchNorm2d(Curvature):

    def update_in_backward(self, grad_output_data):
        pass


class DiagFisherBatchNorm2d(DiagCurvature):

    def update_in_backward(self, grad_out_data):
        input_data = self._input_data  # n x c x h x w

        in_in = input_data.mul(input_data).sum(dim=(2, 3))  # n x c
        grad_grad = grad_out_data.mul(grad_out_data).sum(dim=(2, 3))  # n x c

        data_w = in_in.mul(grad_grad).mean(dim=0)  # c x 1

        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # c x 1
            self._data.append(data_b)

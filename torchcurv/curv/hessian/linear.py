import torch
from torchcurv import KronFisherLinear


class KronHessianLinear(KronFisherLinear):

    def __init__(self, module, ema_decay=1., damping=0,
                 pre_curv=None, post_curv=None, recursive_approx=False):
        super(KronHessianLinear, self).__init__(module, ema_decay, damping,
                                                pre_curv=pre_curv, post_curv=post_curv)
        self.recursive_approx = recursive_approx

    def forward_postprocess(self, module, input, output):
        super(KronHessianLinear, self).forward_postprocess(module, input, output)
        setattr(self._module, 'output', output)

    def backward_postprocess(self, module, grad_input, grad_output):
        index = 1 if self.bias else 0
        setattr(self._module, 'grad_input', grad_input[index])
        output = getattr(self._module, 'output', None)
        if output is None:
            return

        grad_output = grad_output[0]
        n, dim = grad_output.size()
        post_curv = self.post_curv

        if post_curv is not None:
            post_module = post_curv.module
            post_hessian_input = getattr(post_module, 'hessian_input', None)  # n x f_out x f_out
            post_grad_input = getattr(post_module, 'grad_input', None)  # n x f_out

            msg = 'sample hessian/grad of loss w.r.t. inputs of post layer' \
                  ' has to be computed beforehand.'
            assert post_hessian_input is not None, msg
            assert post_grad_input is not None, msg

            recursive_approx = getattr(post_curv, 'recursive_approx', False)

            # compute grad of post activation (we support piece wise activation only)
            zero_indices = [grad_output == 0]  # for avoiding division by zero
            grad_activation = post_grad_input.div(grad_output)  # n x f_out
            grad_activation[zero_indices] = 0
            grad_activation_2d = torch.diag_embed(grad_activation)  # n x f_out x f_out

            # compute sample hessian_output based on hessian_input of post module
            hessian_output = torch.einsum('bij,bjk,bkl->bil',  # n x f_out x f_out
                                          grad_activation_2d,
                                          post_hessian_input,
                                          grad_activation_2d)

        else:
            # compute sample hessian_output from scratch
            hessian_output = torch.zeros((n, dim, dim))  # n x f_out x f_out
            for i in range(dim):
                outputs = tuple(g[i] for g in grad_output)
                inputs = output
                grad = torch.autograd.grad(outputs, inputs, create_graph=True)
                hessian_output[:, i, :] = grad[0]

        device = grad_output.device
        hessian_output = hessian_output.to(device)
        setattr(self._module, 'hessian_output', hessian_output)

        # refresh hessian_output
        self._G = hessian_output.sum(dim=(0,))

        if self.pre_curv is not None:
            # compute sample hessian_input
            w = self._module.weight.data  # f_out x f_in
            hessian_input = torch.einsum('ij,bjk,kl->bil', w.t(), hessian_output, w)  # n x f_in x f_in
            setattr(self._module, 'hessian_input', hessian_input)


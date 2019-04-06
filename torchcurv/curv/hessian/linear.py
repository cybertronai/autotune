import torch
from torchcurv import KronFisherLinear


class KronHessianLinear(KronFisherLinear):

    def __init__(self, module, ema_decay=1., damping=0,
                 pre_curv=None, post_curv=None, recursive_approx=False):
        super(KronHessianLinear, self).__init__(module, ema_decay, damping,
                                                pre_curv=pre_curv, post_curv=post_curv)
        self.recursive_approx = recursive_approx

    def update_in_backward(self, grad_output):
        output = getattr(self._module, 'output', None)
        if output is None:
            return

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

            # compute grad of post activation (we support piece wise activation only)
            zero_indices = [grad_output == 0]  # for avoiding division by zero
            grad_activation = post_grad_input.div(grad_output)  # n x f_out
            grad_activation[zero_indices] = 0
            grad_activation_2d = torch.diag_embed(grad_activation)  # n x f_out x f_out

            recursive_approx = getattr(post_curv, 'recursive_approx', False)

            # compute sample hessian_output based on hessian_input of post module
            equation = 'bij,jk,bkl->bil' if recursive_approx else 'bij,bjk,bkl->bil'
            hessian_output = torch.einsum(equation,  # n x f_out x f_out
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
            if self.recursive_approx:
                hessian_input = torch.einsum('ij,jk,kl->il', w.t(), self._G, w)  # n x f_in x f_in
            else:
                hessian_input = torch.einsum('ij,bjk,kl->bil', w.t(), hessian_output, w)  # n x f_in x f_in
            setattr(self._module, 'hessian_input', hessian_input)


from .fisher_block import LinearFB, Conv2dFB
import torch
from collections import defaultdict, Iterable
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer
from torchcurv.optim import SecondOrderOptimizer


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, curv_type, lr=0.01, momentum=0.9, momentum_type='precgrad', adjust_momentum=False, l2_reg=0, weight_decay=0, num_samples=10, std_scale=4e-6, **curv_kwargs):
        super(VIOptimizer, self).__init__(model, curv_type, lr=0.01, momentum=0.9,
                                          momentum_type='precgrad', adjust_momentum=False, l2_reg=0, weight_decay=0, **curv_kwargs)
        self.defaults['num_samples'] = num_samples
        self.defaults['std_scale'] = std_scale
        self.fisher_init = False

    def step(self):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        '''
        def closure():
            # forward/backward
            return loss, outputs
        '''

        # initialize fisher matrix (for only init))
        if self.fisher_init is False:
            closure()  # forward
            for group in self.param_groups:
                curv = group['curv']
                if curv is not None:
                    curv.update_ema()
                    curv.update_inv()
                    curv.update_std()
                    group['mean'] = [p.clone().detach()
                                     for p in group['params']]
                    group['mean_grad'] = [torch.zeros_like(
                        p, device=p.device) for p in group['params']]
            self.fisher_init = True

        # copy params to mean & fill mean_grad with 0
        for group in self.param_groups:
            for p, m, m_grad in zip(params, mean, m_grad):
                m.copy_(p.data)
                m_grad.fill_(0)

        # sampling and buf update
        n = self.defaults['num_samples']
        std_scale = self.defaults['std_scale']
        loss_avg = None
        outputs_avg = None
        for i in range(n):

            # sampling
            for group in self.param_groups:
                params, mean, curv = group['params'], group['mean'], group['curv']
                curv.sample_param(params, mean, std_scale)

            # forward and backward (curv.data is accumulated)
            # TODO curv accumulate(for vi)
            loss, outputs = closure()

            if loss_avg is None:
                loss_avg = loss.data.mul(1/n)
                outputs_avg = outputs.data.mul(1/n)
            else:
                loss_avg.add_(1/n, loss.data)
                outputs_avg.add_(1/n, outputs.data)

            # update buf
            for group in self.param_groups:
                params = group['params']
                mean_grad = group['mean_grad']
                curv = group['curv']
                if curv is not None:
                    for p, m_grad in zip(params, mean_grad):

                        if p.grad is None:
                            continue

                        grad = p.grad.data

                        if group['l2_reg'] != 0:
                            if grad.is_sparse:
                                raise RuntimeError(
                                    "l2 regularization option is not compatible with sparse gradients")
                            grad.add_(group['l2_reg'], p.data)

                        m_grad.add_(1/n, grad)

        # update distribution and clear buf
        for group in self.param_groups:
            params = group['params']
            mean = group['mean']
            mean_grad = group['mean_grad']
            curv = group['curv']
            if curv is not None:
                for p, m, m_grad in zip(params, mean, m_grad):

                    if p.grad is None:
                        continue

                    p.data.copy_(m)
                    p.grad.data.copy_(m_grad)

                    grad = p.grad.data

                    if group['momentum_type'] == 'grad':
                        momentum = self.momentum(group)
                        self.apply_momentum(p, grad, momentum)

                curv.update_ema()
                curv.update_inv()
                precgrad = curv.precgrad(params)

                for p, grad in zip(params, precgrad):

                    if group['weight_decay'] != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients")
                        grad.add_(group['weight_decay'], p.data)

                    if group['momentum_type'] == 'precgrad':
                        momentum = self.momentum(group)
                        self.apply_momentum(p, grad, momentum)

                    p.data.add_(-group['lr'], grad)

        return outputs_avg, loss_avg

import torch
from collections import defaultdict, Iterable
from torchcurv.optim import SecondOrderOptimizer
import torch.nn.functional as F


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, curv_type, lr=0.01, momentum=0.9, momentum_type='precgrad', adjust_momentum=False, l2_reg=0, weight_decay=0, num_samples=10, std_scale=4e-6, **curv_kwargs):
        super(VIOptimizer, self).__init__(model, curv_type, lr=lr, momentum=momentum,
                                          momentum_type=momentum_type, adjust_momentum=adjust_momentum, l2_reg=l2_reg, weight_decay=weight_decay, **curv_kwargs)
        self.defaults['num_samples'] = num_samples
        self.defaults['std_scale'] = std_scale
        self.fisher_init = False

    def update_preprocess(self, group):
        params = group['params']
        mean = group['mean']
        mean_grad = group['mean_grad']
        for p, m, m_grad in zip(params, mean, mean_grad):

            if p.grad is None:
                continue

            p.data.copy_(m)
            p.grad.data.copy_(m_grad)

            grad = p.grad.data

            if group['momentum_type'] == 'grad':
                momentum = self.momentum(group)
                self.apply_momentum(p, grad, momentum)


    def closure(self):
        data, target = self.data, self.target
        self.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        return loss, output

    def step(self, data=None, target=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        '''
        def closure():
            # forward/backward
            return loss, output
        '''
        self.data = data
        self.target = target
        if closure is None:
            if data is None or target is None:
                raise RuntimeError('VIOptimizer needs closure or data and target')
            closure = self.closure

        # initialize fisher matrix (for only init))
        if self.fisher_init is False:
            loss, _ = closure()  # forward/backward
            for group in self.param_groups:
                curv = group['curv']
                if curv is not None:
                    curv.update_ema(1)
                    curv.update_inv()
                    curv.update_std()
                    group['mean'] = [p.clone().detach()
                                     for p in group['params']]
                    group['mean_grad'] = [torch.zeros_like(
                        p, device=p.device) for p in group['params']]
            self.fisher_init = True

        # copy params to mean & fill mean_grad with 0
        for group in self.param_groups:
            params = group['params']
            mean = group['mean']
            mean_grad = group['mean_grad']
            for p, m, m_grad in zip(params, mean, mean_grad):
                m.copy_(p.data)
                m_grad.fill_(0)

        # sampling and buf update
        n = self.defaults['num_samples']
        std_scale = self.defaults['std_scale']
        loss_avg = None
        output_avg = None
        for i in range(n):

            # sampling
            for group in self.param_groups:
                params, mean, curv = group['params'], group['mean'], group['curv']
                curv.sample_params(params, mean, std_scale)

            # forward and backward (curv.data is accumulated)
            loss, output = closure()

            if loss_avg is None:
                loss_avg = loss.data.mul(1/n)
                output_avg = F.softmax(output, dim=1).data.mul(1/n)
            else:
                loss_avg.add_(1/n, loss.data)
                output_avg.add_(1/n, F.softmax(output, dim=1).data)

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
            curv = group['curv']
            if curv is not None:
                self.update_preprocess(group)

                curv.update_ema(n)
                curv.update_inv()
                curv.update_std()
                precgrad = curv.precgrad(params)

                self.update_postprocess(group, precgrad)
        return loss_avg, output_avg

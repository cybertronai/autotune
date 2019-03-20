import math

import torch
from torchcurv.optim import SecondOrderOptimizer
from torchcurv.utils import TensorAccumulator


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, dataset_size, curv_type, curv_shapes,
                 lr=0.01, momentum=0.9, momentum_type='precgrad', adjust_momentum=False, weight_decay=0,
                 num_mc_samples=10, kl_weighting=0.2, prior_variance=1., **curv_kwargs):

        l2_reg = kl_weighting / dataset_size / prior_variance if prior_variance != 0 else 0

        super(VIOptimizer, self).__init__(model, curv_type, curv_shapes, lr=lr, momentum=momentum,
                                          momentum_type=momentum_type, adjust_momentum=adjust_momentum,
                                          l2_reg=l2_reg, weight_decay=weight_decay, **curv_kwargs)

        self.defaults['num_mc_samples'] = num_mc_samples
        self.defaults['std_scale'] = math.sqrt(kl_weighting / dataset_size)

        self.state['step'] = 0

        for group in self.param_groups:
            group['mean'] = [p.clone().detach() for p in group['params']]
            for m in group['mean']:
                state = self.state[m]
                state['momentum_buffer'] = torch.zeros_like(m.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        def closure():
            # forward/backward
            return loss, output
        """

        n = self.defaults['num_mc_samples'] if self.state['step'] > 0 else 1
        acc_loss = TensorAccumulator()
        acc_output = TensorAccumulator()

        for i in range(n):

            # sampling
            for group in self.param_groups:
                params, mean = group['params'], group['mean']
                curv = group['curv']
                if curv is not None and curv.std is not None:
                    curv.sample_params(params, mean, self.defaults['std_scale'])
                else:
                    for p, m in zip(params, mean):
                        p.data.copy_(m.data)

            # forward and backward
            loss, output = closure()

            acc_loss.update(loss, scale=1/n)
            acc_output.update(output, scale=1/n)

            # update buf
            for group in self.param_groups:
                params = group['params']

                grads = [p.grad.data for p in params]
                group['acc_grads'].update(grads, scale=1/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/n)

        self.state['step'] += 1

        # update distribution
        for group in self.param_groups:
            mean = group['mean']

            # update mean grad
            acc_grads = group['acc_grads'].get()
            for m, acc_grad in zip(mean, acc_grads):
                m.grad = acc_grad.clone()
            self.update_preprocess(group, target='mean', attr='grad')

            curv = group['curv']
            if curv is not None:
                # update covariance
                curv.data = group['acc_curv'].get()
                curv.update_ema()
                curv.update_inv()
                curv.update_std()
                curv.precondition_grad(mean)

            # update mean
            self.update_preprocess(group, target='mean', attr='precgrad')
            self.update(group, target='mean')

        loss, output = acc_loss.get(), acc_output.get()

        return loss, output


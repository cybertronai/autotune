import math

import torch
from torchcurv.optim import SecondOrderOptimizer
from torchcurv.utils import TensorAccumulator


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, dataset_size, curv_type='Fisher', curv_shapes=None,
                 lr=0.01, momentum=0, momentum_type='preconditioned', adjust_momentum=False,
                 grad_ema_decay=1, grad_ema_type='raw', weight_decay=0,
                 num_mc_samples=10, test_num_mc_samples=10, kl_weighting=1, prior_variance=1,
                 **curv_kwargs):

        l2_reg = kl_weighting / dataset_size / prior_variance if prior_variance != 0 else 0

        super(VIOptimizer, self).__init__(model, curv_type, curv_shapes, lr=lr, momentum=momentum,
                                          momentum_type=momentum_type, adjust_momentum=adjust_momentum,
                                          grad_ema_decay=grad_ema_decay, grad_ema_type=grad_ema_type,
                                          l2_reg=l2_reg, weight_decay=weight_decay, **curv_kwargs)

        self.defaults['num_mc_samples'] = num_mc_samples
        self.defaults['test_num_mc_samples'] = test_num_mc_samples
        self.defaults['std_scale'] = math.sqrt(kl_weighting / dataset_size)
        self.defaults['prior_variance'] = prior_variance

        for group in self.param_groups:
            group['mean'] = [torch.zeros_like(p) for p in group['params']]
            self.init_buffer(group['mean'])

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for m in group['mean']:
                if m.grad is not None:
                    m.grad.detach_()
                    m.grad.zero_()

        super(VIOptimizer, self).zero_grad()

    def set_random_seed_by_step(self):
        seed = self.optim_state['step']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_params(self):
        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            curv = group['curv']
            if curv is not None and curv.std is not None:
                # sample from posterior
                curv.sample_params(params, mean, self.defaults['std_scale'])
            else:
                # sample from prior
                for p, m in zip(params, mean):
                    noise = torch.randn_like(m)
                    std = noise.mul_(math.sqrt(self.defaults['prior_variance']))
                    p.data.copy_(torch.add(m, std))

    def set_mean_to_params(self):
        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            for p, m in zip(params, mean):
                p.data.copy_(m.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        def closure():
            # forward/backward
            return loss, output
        """

        m = self.defaults['num_mc_samples']
        n = self.defaults['acc_steps']

        acc_loss = TensorAccumulator()
        acc_output = TensorAccumulator()

        self.set_random_seed_by_step()

        for _ in range(m):

            # sampling
            self.sample_params()

            # forward and backward
            loss, output = closure()

            acc_loss.update(loss, scale=1/m)
            acc_output.update(output, scale=1/m)

            # accumulate
            for group in self.param_groups:
                params = group['params']

                grads = [p.grad.data for p in params]
                group['acc_grads'].update(grads, scale=1/m/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/m/n)

        loss, output = acc_loss.get(), acc_output.get()

        # update acc step
        self.optim_state['acc_step'] += 1
        if self.optim_state['acc_step'] < n:
            return loss, output
        else:
            self.optim_state['acc_step'] = 0

        self.backward_postprocess(target='mean')
        self.optim_state['step'] += 1

        # update distribution
        for group in self.local_param_groups:

            self.update_preprocess(group, target='mean', grad_type='raw')

            # update covariance
            mean, curv = group['mean'], group['curv']
            if curv is not None:
                curv.step(update_std=True)
                curv.precondition_grad(mean)

            # update mean
            self.update_preprocess(group, target='mean', grad_type='preconditioned')
            self.update(group, target='mean')

        # set mean to model.params
        self.set_mean_to_params()

        return loss, output

    def prediction(self, data):

        acc_output = TensorAccumulator()
        mc_samples = self.defaults['test_num_mc_samples']

        use_mean = mc_samples == 0
        n = 1 if use_mean else mc_samples

        for _ in range(n):

            if not use_mean:
                # sampling
                self.sample_params()

            output = self.model(data)
            acc_output.update(output, scale=1/n)

        # set mean to model.params
        self.set_mean_to_params()

        output = acc_output.get()

        return output


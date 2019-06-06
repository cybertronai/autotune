import math

import torch
import torch.nn.functional as F
from torchcurv.optim import SecondOrderOptimizer, DistributedSecondOrderOptimizer
from torchcurv.utils import TensorAccumulator
from torchcurv.utils.chainer_communicators import _utility


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, dataset_size, curv_type='Fisher', curv_shapes=None,
                 lr=0.01, momentum=0, momentum_type='preconditioned',
                 grad_ema_decay=1, grad_ema_type='raw', weight_decay=0,
                 normalizing_weights=False, weight_scale='auto',
                 acc_steps=1, non_reg_for_bn=False, bias_correction=False,
                 lars=False, lars_type='preconditioned',
                 num_mc_samples=10, val_num_mc_samples=10,
                 kl_weighting=1, warmup_kl_weighting_init=0.01, warmup_kl_weighting_steps=None,
                 prior_variance=1, init_precision=None,
                 seed=1, total_steps=1000, **curv_kwargs):

        init_kl_weighting = kl_weighting if warmup_kl_weighting_steps is None else warmup_kl_weighting_init
        l2_reg = init_kl_weighting / dataset_size / prior_variance if prior_variance != 0 else 0
        std_scale = math.sqrt(init_kl_weighting / dataset_size)

        super(VIOptimizer, self).__init__(model, curv_type, curv_shapes,
                                          lr=lr, momentum=momentum, momentum_type=momentum_type,
                                          grad_ema_decay=grad_ema_decay, grad_ema_type=grad_ema_type,
                                          l2_reg=l2_reg, weight_decay=weight_decay,
                                          normalizing_weights=normalizing_weights, weight_scale=weight_scale,
                                          acc_steps=acc_steps, non_reg_for_bn=non_reg_for_bn,
                                          bias_correction=bias_correction,
                                          lars=lars, lars_type=lars_type,
                                          **curv_kwargs)

        self.defaults['std_scale'] = std_scale
        self.defaults['kl_weighting'] = kl_weighting
        self.defaults['warmup_kl_weighting_init'] = warmup_kl_weighting_init
        self.defaults['warmup_kl_weighting_steps'] = warmup_kl_weighting_steps
        self.defaults['num_mc_samples'] = num_mc_samples
        self.defaults['val_num_mc_samples'] = val_num_mc_samples
        self.defaults['total_steps'] = total_steps
        self.defaults['seed_base'] = seed

        for group in self.param_groups:
            group['std_scale'] = 0 if group['l2_reg'] == 0 else std_scale
            group['mean'] = [p.data.detach().clone() for p in group['params']]
            self.init_buffer(group['mean'])

            if init_precision is not None:
                curv = group['curv']
                curv.element_wise_init(init_precision)
                curv.step(update_std=(group['std_scale'] > 0))

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for m in group['mean']:
                if m.grad is not None:
                    m.grad.detach_()
                    m.grad.zero_()

        super(VIOptimizer, self).zero_grad()

    @property
    def seed(self):
        return self.optim_state['step'] + self.defaults['seed_base']

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_params(self):

        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            curv = group['curv']
            if curv is not None and curv.std is not None:
                # sample from posterior
                curv.sample_params(params, mean, group['std_scale'])
            else:
                for p, m in zip(params, mean):
                    p.data.copy_(m.data)

    def copy_mean_to_params(self):
        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            for p, m in zip(params, mean):
                p.data.copy_(m.data)
                if getattr(p, 'grad', None) is not None \
                        and getattr(m, 'grad', None) is not None:
                    p.grad.copy_(m.grad)

    def adjust_kl_weighting(self):
        warmup_steps = self.defaults['warmup_kl_weighting_steps']
        if warmup_steps is None:
            return

        current_step = self.optim_state['step']
        if warmup_steps < current_step:
            return

        target_kl = self.defaults['kl_weighting']
        init_kl = self.defaults['warmup_kl_weighting_init']

        rate = current_step / warmup_steps
        kl_weighting = init_kl + rate * (target_kl - init_kl)

        rate = kl_weighting / init_kl
        l2_reg = rate * self.defaults['l2_reg']
        std_scale = math.sqrt(rate) * self.defaults['std_scale']
        for group in self.param_groups:
            if group['l2_reg'] > 0:
                group['l2_reg'] = l2_reg
            if group['std_scale'] > 0:
                group['std_scale'] = std_scale

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
        acc_prob = TensorAccumulator()

        self.set_random_seed()

        for _ in range(m):

            # sampling
            self.sample_params()

            # forward and backward
            loss, output = closure()

            acc_loss.update(loss, scale=1/m)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1/n)

            # accumulate
            for group in self.param_groups:
                params = group['params']

                grads = [p.grad.data for p in params]
                group['acc_grads'].update(grads, scale=1/m/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/m/n)

        loss, prob = acc_loss.get(), acc_prob.get()

        # update acc step
        self.optim_state['acc_step'] += 1
        if self.optim_state['acc_step'] < n:
            return loss, prob
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
                curv.step(update_std=(group['std_scale'] > 0))
                curv.precondition_grad(mean)

            # update mean
            self.update_preprocess(group, target='mean', grad_type='preconditioned')
            self.update(group, target='mean')
            self.update_postprocess(group, target='mean')

            # copy mean to param
            params = group['params']
            for p, m in zip(params, mean):
                p.data.copy_(m.data)
                p.grad.copy_(m.grad)

        self.adjust_kl_weighting()

        return loss, prob

    def prediction(self, data):

        self.set_random_seed(self.optim_state['step'])

        acc_prob = TensorAccumulator()
        mc_samples = self.defaults['val_num_mc_samples']

        use_mean = mc_samples == 0
        n = 1 if use_mean else mc_samples

        for _ in range(n):

            if use_mean:
                self.copy_mean_to_params()
            else:
                # sampling
                self.sample_params()

            output = self.model(data)
            prob = F.softmax(output, dim=1)
            acc_prob.update(prob, scale=1/n)

        self.copy_mean_to_params()

        prob = acc_prob.get()

        return prob


class DistributedVIOptimizer(DistributedSecondOrderOptimizer, VIOptimizer):

    def __init__(self, *args, mc_group_id=0, **kwargs):
        super(DistributedVIOptimizer, self).__init__(*args, **kwargs)
        self.defaults['seed_base'] += mc_group_id * self.defaults['total_steps']

    @property
    def actual_optimizer(self):
        return VIOptimizer

    def zero_grad(self):
        self.actual_optimizer.zero_grad(self)

    def extractors_for_rsv(self):
        extractors = [_utility.extract_attr_from_params('grad', target='mean'),
                      _utility.extract_attr_from_curv('data', True)]
        return extractors

    def extractors_for_agv(self):
        extractors = [_utility.extract_attr_from_params('data', target='mean'),
                      _utility.extract_attr_from_curv('std', True)]
        return extractors

    def step(self, closure=None):
        ret = super(DistributedVIOptimizer, self).step(closure)

        if self.is_updated():
            self.copy_mean_to_params()

        return ret


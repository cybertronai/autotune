from collections import defaultdict

import torch
from torch.optim import Optimizer
import torchcurv
from torchcurv.utils import TensorAccumulator


class SecondOrderOptimizer(Optimizer):

    def __init__(self, model, curv_type, curv_shapes, lr=0.01,
                 momentum=0.9, momentum_type='precgrad', adjust_momentum=False,
                 l2_reg=0, weight_decay=0, **curv_kwargs):
        # TODO implement error checker: hoge(optim_kwargs)
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= l2_reg:
            raise ValueError("Invalid l2_reg value: {}".format(l2_reg))
        if not 0.0 <= damping:
            raise ValueError("Invalid damping value: {}".format(damping))
        if not 0.0 <= cov_ema_decay:
            raise ValueError(
                "Invalid cov_ema_decay value: {}".format(cov_ema_decay))
        """
        self.model = model
        defaults = {'lr': lr, 'lr_pre': lr, 'momentum': momentum, 'momentum_type': momentum_type,
                    'adjust_momentum': adjust_momentum, 'l2_reg': l2_reg, 'weight_decay': weight_decay}
        defaults.update(curv_kwargs)
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.train_modules = []
        self.set_train_modules(model)  # TODO implement better method
        self.param_groups = []
        self.curv_type = curv_type
        self.curv_shapes = curv_shapes

        for module in self.train_modules:
            params = list(module.parameters())
            curv_class = self.get_curv_class(module)
            if curv_class is not None:
                curvature = curv_class(module, **curv_kwargs)
            else:
                curvature = None

            group = {
                'params': params,
                'curv': curvature,
                'acc_curv': TensorAccumulator(),
                'acc_grads': TensorAccumulator()
            }

            self.add_param_group(group)

            for p in params:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)

        exit()

    def get_curv_class(self, module):
        module_name = module.__class__.__name__
        curv_shape = self.curv_shapes.get(module_name, '')
        curv_name = curv_shape + self.curv_type + module_name
        curv_class = getattr(torchcurv, curv_name, None)
        print(curv_class)

        return curv_class

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        self.model.zero_grad()

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def apply_momentum(self, p, grad, momentum):
        if momentum != 0:
            state = self.state[p]
            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad)
            grad.copy_(buf)

    def momentum(self, group):
        if group['adjust_momentum']:
            lr, lr_pre, m = group['lr'], group['lr_pre'], group['momentum']
            return m/lr_pre*lr
        else:
            return group['momentum']

    def update_preprocess(self, group, target='params', attr='grad'):
        params = group[target]

        for p in params:

            grad = getattr(p, attr, p.grad)

            if grad is None:
                continue

            if attr == 'grad':
                if group['l2_reg'] != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "l2 regularization option is not compatible with sparse gradients")
                    grad.add_(group['l2_reg'], p.data)

            if attr == 'precgrad':
                if group['weight_decay'] != 0:
                    if hasattr(grad, 'is_sparse') and grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients")
                    grad.add_(group['weight_decay'], p.data)

            if group['momentum_type'] == attr:

                momentum = self.momentum(group)
                self.apply_momentum(p, grad, momentum)

    def update(self, group, target='params'):
        params = group[target]

        for p in params:

            grad = p.precgrad if hasattr(p, 'precgrad') else p.grad

            if grad is None:
                continue

            p.data.add_(-group['lr'], grad)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params = group['params']

            self.update_preprocess(group, attr='grad')

            curv = group['curv']
            if curv is not None:
                curv.update_ema()
                curv.update_inv()
                curv.precondition_grad(params)

            self.update_preprocess(group, attr='precgrad')
            self.update(group)

        return loss

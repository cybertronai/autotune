import torch
from collections import defaultdict
import torch.nn as nn
from torch.optim import Optimizer
import torchcurv.curv as curv
import inspect


def get_curv_class(curv_type, module):
    # TODO implement
    if isinstance(module, nn.Linear):
        module_type = 'Linear'
    elif isinstance(module, nn.Conv2d):
        module_type = 'Conv2d'
    elif isinstance(module, nn.BatchNorm2d):
        #module_type = 'BatchNorm2d'
        return None
    else:
        return None

    curv_class = getattr(curv, curv_type+module_type)

    return curv_class


def extract_kwargs(func, target):
    if target is None:
        return {}

    keys = list(inspect.signature(func).parameters.keys())
    kwargs = {}
    for key, val in target.items():
        if key in keys:
            kwargs[key] = val
    return kwargs


class SecondOrderOptimizer(Optimizer):

    def __init__(self, model, curv_type, lr=0.01, momentum=0.9, l2_reg=0, weight_decay=0, **curv_kwargs):
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
        defaults = {'lr': lr, 'momentum': momentum,
                    'l2_reg': l2_reg, 'weight_decay': weight_decay}
        defaults.update(curv_kwargs)
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.train_modules = []
        self.set_train_modules(model)  # TODO implement better method
        self.param_groups = []

        for module in self.train_modules:
            params = list(module.parameters())
            curv_class = get_curv_class(curv_type, module)
            if curv_class is not None:
                curvature = curv_class(module, **curv_kwargs)
            else:
                curvature = None
            group = {
                'params': params,
                'curv': curvature
            }
            self.add_param_group(group)
            for p in params:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)

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
            curv = group['curv']
            if curv is not None:
                for p in params:
                    if p.grad is None:
                        continue

                    grad = p.grad.data

                    if group['l2_reg'] != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "l2 regularization option is not compatible with sparse gradients")
                        grad.add_(group['l2_reg'], p.data)

                precgrad = curv.compute_precgrad(params)
                for p, grad in zip(params, precgrad):
                    if group['weight_decay'] != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients")
                        grad.add_(group['weight_decay'], p.data)
                    momentum = group['momentum']
                    if momentum != 0:
                        state = self.state[p]
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                        v = buf
                    else:
                        v = grad
                    p.data.add_(-group['lr'], v)
        return loss

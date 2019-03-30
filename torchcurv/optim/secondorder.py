from collections import defaultdict
import inspect

import torch
from torch.optim import Optimizer
import torchcurv
from torchcurv.utils import TensorAccumulator

from torchcurv.utils.chainer_communicators import create_communicator
import numpy as np


class SecondOrderOptimizer(Optimizer):

    def __init__(self, model, curv_type, curv_shapes,
                 lr=0.01, momentum=0, momentum_type='preconditioned', adjust_momentum=False,
                 grad_ema_decay=1, grad_ema_type='raw', l2_reg=0, weight_decay=0, acc_steps=1,
                 **curv_kwargs):

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
        defaults = {'lr': lr, 'lr_pre': lr,
                    'momentum': momentum, 'momentum_type': momentum_type, 'adjust_momentum': adjust_momentum,
                    'grad_ema_decay': grad_ema_decay, 'grad_ema_type': grad_ema_type,
                    'l2_reg': l2_reg, 'weight_decay': weight_decay, 'acc_steps': acc_steps}
        defaults.update(curv_kwargs)
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.optim_state = {'step': 0, 'acc_step': 0}

        self.train_modules = []
        self.set_train_modules(model)  # TODO implement better method
        self.param_groups = []
        self.curv_type = curv_type
        self.curv_shapes = {} if curv_shapes is None else curv_shapes

        def extract_kwargs(func, target):
            if target is None:
                return {}

            keys = list(inspect.signature(func).parameters.keys())
            kwargs = {}
            for key, val in target.items():
                if key in keys:
                    kwargs[key] = val
            return kwargs

        for module in self.train_modules:
            params = list(module.parameters())
            curv_class = self.get_curv_class(module)
            if curv_class is not None:
                kwargs = extract_kwargs(curv_class.__init__, curv_kwargs)
                curvature = curv_class(module, **kwargs)
            else:
                curvature = None

            group = {
                'name': module.__class__.__name__,
                'params': params,
                'curv': curvature,
                'acc_curv': TensorAccumulator(),
                'acc_grads': TensorAccumulator()
            }

            self.add_param_group(group)

            for p in params:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['grad_ema_buffer'] = torch.zeros_like(p.data)

    @property
    def local_param_groups(self):
        return self.param_groups

    def get_curv_class(self, module):
        module_name = module.__class__.__name__
        curv_shape = self.curv_shapes.get(module_name, '')
        curv_name = curv_shape + self.curv_type + module_name
        curv_class = getattr(torchcurv, curv_name, None)

        return curv_class

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
        n = self.defaults['acc_steps']

        if closure is not None:
            loss = closure()

            for group in self.param_groups:
                params = group['params']

                grads = [p.grad.data for p in params]
                group['acc_grads'].update(grads, scale=1/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/n)

            self.optim_state['acc_step'] += 1
            if self.optim_state['acc_step'] < n:
                return loss
            else:
                self.optim_state['acc_step'] = 0

            self.backward_postprocess()

        self.optim_state['step'] += 1

        for group in self.local_param_groups:
            params = group['params']

            self.update_preprocess(group, grad_type='raw')

            curv = group['curv']
            if curv is not None:
                curv.step()
                curv.precondition_grad(params)

            self.update_preprocess(group, grad_type='preconditioned')
            self.update(group)

        return loss

    def backward_postprocess(self):
        for group in self.param_groups:
            params = group['params']

            acc_grads = group['acc_grads'].get()
            for p, acc_grad in zip(params, acc_grads):
                p.grad = acc_grad.clone()

            curv = group['curv']
            if curv is not None:
                curv.data = group['acc_curv'].get()

    def update(self, group, target='params'):
        params = group[target]

        for p in params:

            grad = p.grad

            if grad is None:
                continue

            p.data.add_(-group['lr'], grad)

    def update_preprocess(self, group, target='params', grad_type='raw'):
        assert grad_type in ['raw', 'preconditioned'], 'Invalid grad type: {}.'.format(grad_type)

        params = group[target]

        for p in params:

            grad = p.grad

            if grad is None:
                continue

            if grad_type == 'raw':
                self.apply_l2_reg(p, grad, group)

            if grad_type == 'preconditioned':
                self.apply_weight_decay(p, grad, group)

            if group['momentum_type'] == grad_type:
                self.apply_momentum(p, grad, group)

            if group['grad_ema_type'] == grad_type:
                self.apply_grad_ema_decay(p, grad, group)

    def apply_l2_reg(self, p, grad, group):
        if group['l2_reg'] != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "l2 regularization option is not compatible with sparse gradients")
            grad.add_(group['l2_reg'], p.data)
            curv = group['curv']
            if curv is not None:
                curv.l2_reg = group['l2_reg']

    def apply_weight_decay(self, p, grad, group):
        if group['weight_decay'] != 0:
            if hasattr(grad, 'is_sparse') and grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients")
            grad.add_(group['weight_decay'], p.data)

    def apply_momentum(self, p, grad, group):
        if group['adjust_momentum']:
            lr, lr_pre, m = group['lr'], group['lr_pre'], group['momentum']
            momentum = m/lr_pre*lr
        else:
            momentum = group['momentum']

        if momentum != 0:
            state = self.state[p]
            buf = state['momentum_buffer']
            buf.mul_(momentum).add_(grad)
            grad.copy_(buf)

    def apply_grad_ema_decay(self, p, grad, group):
        grad_ema_decay = group['grad_ema_decay']
        if grad_ema_decay != 1:
            state = self.state[p]
            buf = state['grad_ema_buffer']
            buf.mul_(1 - grad_ema_decay).add_(grad.mul(grad_ema_decay))
            grad.copy_(buf)


class DistributedSecondOrderOptimizer(SecondOrderOptimizer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.comm = create_communicator()

        local_size = self.comm.size
        local_rank = self.comm.rank
        indices = np.array_split(np.arange(len(self.param_groups)), local_size)
        indices = [local_indices.tolist() for local_indices in indices]
        local_indices = indices[local_rank]
        local_param_groups = [self.param_groups[i] for i in local_indices]

        self.indices = indices
        self.local_indices = local_indices
        self._local_param_groups = local_param_groups
        setattr(self.comm, 'indices', indices)

    @property
    def local_param_groups(self):
        return self._local_param_groups

    def backward_postprocess(self):
        super().backward_postprocess()
        # reduce_scatterv
        self.comm.reduce_scatterv_data(self.param_groups)

    def is_updated(self):
        return self.optim_state['acc_step'] == 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = super().step(closure)

        if self.is_updated():
            # allgatherv
            self.comm.allgatherv_data(self.param_groups)

        return loss


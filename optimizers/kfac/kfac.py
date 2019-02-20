import torch
from torch.optim import Optimizer
import torch.nn as nn

from .fisher_block import LinearFB,Conv2dFB

from collections import defaultdict, Iterable

_default_lr = 1e-2
_default_momentum = 0.9
_default_weight_decay = 0
_default_l2_reg = 0
_default_damping = 0.01
_default_cov_ema_decay = 0.99
_default_pi_type = 'trace_norm'

required = object()

def update_input(self,input,output):
    self.input = input[0].data


def update_grad_output(self,grad_input,grad_output):
    self.grad_output = grad_output[0].data

class KFAC(Optimizer):
    """Implements KFAC algorithm.

    It has been proposed in `Optimizing Neural Networks with \
            Kronecker-factored Approximate Curvature. \
            <https://arxiv.org/abs/1503.05671>`_

    Arguments:
        model (nn.Module): network model 
        lr (float, optional): learning rate 
        momentum (float, optional): adaptive momentum 
        weight_decay (float, optional): weight decay 
        l2_reg (float, optional): L2 regularization 
        damping (float, optional): damping for kfac 
        cov_ema_decay (float, optional): exponential moving average of covariance rate 
        pi_type (str,optional): type of pi norm for damping 
    """

    def __init__(self, model,
            lr = _default_lr,
            momentum = _default_momentum,
            weight_decay = _default_weight_decay,
            l2_reg = _default_l2_reg,
            damping = _default_damping,
            cov_ema_decay = _default_cov_ema_decay,
            pi_type = _default_pi_type
            ):
        if not 0.0 <=  lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= l2_reg:
            raise ValueError("Invalid l2_reg value: {}".format(l2_reg))
        if not 0.0 <= damping:
            raise ValueError("Invalid damping value: {}".format(damping))
        if not 0.0 <= cov_ema_decay:
            raise ValueError("Invalid cov_ema_decay value: {}".format(cov_ema_decay))

        defaults = dict(
                lr = lr,
                momentum = momentum,
                weight_decay_base = weight_decay/lr,
                l2_reg = l2_reg,
                damping = damping,
                cov_ema_decay = cov_ema_decay,
                pi_type = pi_type,
                adaptive_momentum_base = momentum/lr
                )
        self.defaults = defaults


        self.train_modules = []
        self.set_train_modules(model)

        param_groups = []
        self.fblocks = []

        for module in self.train_modules:
            module.register_forward_hook(update_input)
            module.register_backward_hook(update_grad_output)
            params = tuple(module.parameters())
            param_groups.append(params)
            bias = True if module.bias is not None else False
            if isinstance(module,nn.Linear):
                self.fblocks.append(LinearFB(damping=damping,cov_ema_decay=cov_ema_decay,bias=bias,pi_type=pi_type))
            elif isinstance(module,nn.Conv2d):
                self.fblocks.append(Conv2dFB(kernel_size=module.kernel_size,stride=module.stride,
                    padding=module.padding,dilation=module.dilation,damping=damping,cov_ema_decay=cov_ema_decay,bias=bias,pi_type=pi_type))
            else:
                self.fblocks.append(None)
        '''
        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))
        '''
        self.state = defaultdict(dict)
        self.param_groups = []

        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        for group in self.param_groups:
            for tp in group['params']:
                for p in tp:    
                    state = self.state[p]
                    state['step'] = 0


    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        tp_group = param_group['params']
        if isinstance(tp_group, tuple):
            param_group['params'] = [tp_group]
        elif isinstance(tp_group, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                    'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(tp_group)

        for tp in param_group['params']:
            if not isinstance(tp,tuple):
                raise TypeError("KFAC use tuple of params, "
                        "but one of the param_group is " + torch.typename(param))
                for param in tp:
                    if not param.is_leaf:
                        raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                        name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for tp in group['params']:
                for p in tp:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_() 
    def set_train_modules(self,module):
        """set modules which have params."""
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
        kfac_bufs = []
        for module in self.train_modules:
            if isinstance(module,nn.Linear) or isinstance(module,nn.Conv2d):
                kfac_buf = (module.input,module.grad_output)
                kfac_bufs.append(kfac_buf)
            else:
                kfac_bufs.append(None)

        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for tp,fb,kfac_buf in zip(group['params'],self.fblocks,kfac_bufs):
                for p in tp:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['step'] += 1

                    grad = p.grad.data

                    if group['l2_reg'] != 0:
                        if p.grad.data.is_sparse:
                            raise RuntimeError("l2 regularization option is not compatible with sparse gradients")
                        grad.add_(group['l2_reg'], p.data)


                if kfac_buf is None:
                    for p in tp:
                        pass
                else:
                    kfgrad = fb(tp,kfac_buf)
                    for p,grad in zip(tp,kfgrad):
                        if group['weight_decay_base'] != 0:
                            if p.grad.data.is_sparse:
                                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                            grad.add_(group['weight_decay_base'], p.data)
                        v = grad
                        momentum = group['adaptive_momentum_base']*group['lr']
                        if momentum != 0:
                            state = self.state[p]
                            if 'momentum_buffer' not in state:
                                buf = state['momentum_buffer'] = torch.zeros_like(p.data)
                                buf.mul_(momentum).add_(grad)
                            else:
                                buf = state['momentum_buffer']
                                buf.mul_(momentum).add_(grad)
                            v = buf
                        p.data.add_(-group['lr'],v)
        return loss 


import torch
import torch.nn.functional as F

from torchcurv.utils import TensorAccumulator


class Fisher(object):

    def __init__(self):
        self.prob = None
        self._do_backward = True
        self._acc_cov = TensorAccumulator()

    @property
    def do_backward(self):
        return self._do_backward

    def turn_on_backward(self):
        self._do_backward = True

    def turn_off_backward(self):
        self._do_backward = False

    def accumulate_cov(self, cov):
        self._acc_cov.update(cov)

    def finalize(self):
        return self._acc_cov.get()

    def update_as_presoftmax(self, prob):
        raise NotImplementedError('This method supports only torchcurv.KronFisherLinear.')


def get_closure_for_fisher(optimizer, model, data, target, approx_type=None, num_mc=1):

    _APPROX_TYPE_MC = 'mc'
    _APPROX_TYPE_RECURSIVE = 'recursive'

    seen_funcs = set()
    all_curvs = [group['curv'] for group in optimizer.param_groups]

    def turn_off_param_grad():
        for group in optimizer.param_groups:
            group['curv'].turn_on_backward()
            for param in group['params']:
                param.requires_grad = False

    def turn_on_param_grad():
        for group in optimizer.param_groups:
            group['curv'].turn_off_backward()
            for param in group['params']:
                param.requires_grad = True

    def get_curvature(variable):
        for curv in all_curvs:
            module = curv.module
            for m in module.parameters(recurse=False):
                if m is variable:
                    return curv
        return None

    def trace_pre_curvatures(func, curv=None):
        if hasattr(func, 'next_functions'):
            next_funcs = set([f[0] for f in func.next_functions])
            next_curv = None
            for _func in next_funcs:
                if hasattr(_func, 'variable'):
                    _curv = get_curvature(_func.variable)
                    next_curv = _curv
                    break

            if next_curv is None:
                next_curv = curv
            else:
                seen_funcs.add(func)
                next_funcs -= seen_funcs

                if curv is not None:
                    _pre_curvs = getattr(curv, 'pre_curvs', set())
                    _pre_curvs.add(next_curv)
                    setattr(curv, 'pre_curvs', _pre_curvs)
                    curv.pre_curvs.discard(curv)
                    _post_curvs = getattr(next_curv, 'post_curvs', set())
                    _post_curvs.add(curv)
                    setattr(next_curv, 'post_curvs', _post_curvs)
                    next_curv.post_curvs.discard(next_curv)

            for _func in next_funcs:
                trace_pre_curvatures(_func, next_curv)

    def closure():

        def cross_entropy(logits, soft_targets):
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            return torch.mean(torch.sum(-soft_targets * logsoftmax(logits), 1))

        for group in optimizer.param_groups:
            assert isinstance(group['curv'], Fisher)

        optimizer.zero_grad()
        output = model(data)
        prob = F.softmax(output, dim=1)

        is_sampling = approx_type is None or approx_type == _APPROX_TYPE_MC

        if is_sampling:
            turn_off_param_grad()

            if approx_type == _APPROX_TYPE_MC:
                dist = torch.distributions.Categorical(prob)
                _target = dist.sample((num_mc,))
                for group in optimizer.param_groups:
                    group['curv'].prob = torch.ones_like(prob[:, 0]).div(num_mc)

                for i in range(num_mc):
                    loss = cross_entropy(output, _target[i])
                    loss.backward(retain_graph=True)
            else:
                for i in range(model.num_classes):
                    for group in optimizer.param_groups:
                        group['curv'].prob = prob[:, i]
                    loss = F.cross_entropy(output, torch.ones_like(target).mul(i))
                    loss.backward(retain_graph=True)

            turn_on_param_grad()

        elif approx_type == _APPROX_TYPE_RECURSIVE:
            trace_pre_curvatures(prob.grad_fn)
            for curv in all_curvs:
                curv.recurse = True
                if len(getattr(curv, 'post_curvs', set())) == 0:
                    curv.update_as_presoftmax(prob)
        else:
            raise ValueError('Invalid approx type: {}'.format(approx_type))

        loss = F.cross_entropy(output, target)
        loss.backward()

        return loss, output

    return closure


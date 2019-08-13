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


def get_closure_for_fisher(optimizer, model, data, target, mc_approx=False, num_mc=1):

    def turn_off_param_grad():
        for group in optimizer.param_groups:
            assert isinstance(group['curv'], Fisher)
            group['curv'].turn_on_backward()
            for param in group['params']:
                param.requires_grad = False

    def turn_on_param_grad():
        for group in optimizer.param_groups:
            group['curv'].turn_off_backward()
            for param in group['params']:
                param.requires_grad = True

    def closure():

        def cross_entropy(logits, soft_targets):
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            return torch.mean(torch.sum(-soft_targets * logsoftmax(logits), 1))

        optimizer.zero_grad()
        output = model(data)
        prob = F.softmax(output, dim=1)

        turn_off_param_grad()

        if mc_approx:
            dist = torch.distributions.Dirichlet(prob)
            soft_target = dist.sample((num_mc,))
            for group in optimizer.param_groups:
                group['curv'].prob = torch.ones_like(prob[:, 0]).div(num_mc)

            for i in range(num_mc):
                loss = cross_entropy(output, soft_target[i])
                loss.backward(retain_graph=True)
        else:
            for i in range(model.num_classes):
                for group in optimizer.param_groups:
                    group['curv'].prob = prob[:, i]
                loss = F.cross_entropy(output, torch.ones_like(target).mul(i))
                loss.backward(retain_graph=True)

        turn_on_param_grad()

        loss = F.cross_entropy(output, target)
        loss.backward()

        return loss, output

    return closure


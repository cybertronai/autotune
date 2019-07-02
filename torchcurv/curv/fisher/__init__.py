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


def get_closure_for_fisher(optimizer, model, data, target):

    def closure():
        optimizer.zero_grad()
        output = model(data)
        prob = F.softmax(output, dim=1)

        for group in optimizer.param_groups:
            assert isinstance(group['curv'], Fisher)
            group['curv'].turn_on_backward()
            for param in group['params']:
                param.requires_grad = False

        for i in range(model.num_classes):
            for group in optimizer.param_groups:
                group['curv'].prob = prob[:, i]
            loss = F.cross_entropy(output, torch.ones_like(target).mul(i))
            loss.backward(retain_graph=True)

        for group in optimizer.param_groups:
            group['curv'].turn_off_backward()
            for param in group['params']:
                param.requires_grad = True

        loss = F.cross_entropy(output, target)
        loss.backward()

        return loss, output

    return closure


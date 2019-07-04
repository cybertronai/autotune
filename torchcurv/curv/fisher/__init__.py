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


def get_closure_for_fisher(optimizer, model, data, target, sample_type=None, num_samples=None):

    def closure():
        optimizer.zero_grad()
        output = model(data)
        prob = F.softmax(output, dim=1)
        n = data.shape[0]

        for group in optimizer.param_groups:
            assert isinstance(group['curv'], Fisher)
            group['curv'].turn_on_backward()
            for param in group['params']:
                param.requires_grad = False

        classes = range(model.num_classes)

        if sample_type is None or num_samples is None:
            stargets = [torch.ones_like(target).mul(i) for i in classes]
            sprob = [prob[:, i] for i in classes]
        else:
            assert num_samples <= model.num_classes
            if sample_type == 'mc':
                sidx = [random.sample(classes, k=num_samples) for _ in range(n)]
                stargets = target.new_tensor(sidx).transpose(0, 1)
                tensors = tuple(p[idx] for p, idx in zip(prob, sidx))
                sprob = torch.stack(tensors).transpose(0, 1)
            elif sample_type == 'topk':
                sprob, stargets = torch.topk(prob, k=num_samples)
                sprob, stargets = sprob.transpose(0, 1), stargets.transpose(0, 1)
            else:
                raise ValueError

        for p, t in zip(sprob, stargets):
            for group in optimizer.param_groups:
                group['curv'].prob = p
            loss = F.cross_entropy(output, t)
            loss.backward(retain_graph=True)

        for group in optimizer.param_groups:
            group['curv'].turn_off_backward()
            for param in group['params']:
                param.requires_grad = True

        loss = F.cross_entropy(output, target)
        loss.backward()

        return loss, output

    return closure


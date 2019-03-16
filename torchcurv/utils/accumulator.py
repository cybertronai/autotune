from torch import Tensor


class TensorAccumulator(object):

    def __init__(self):
        self._accumulation = None

    def update(self, data, scale=1.):
        accumulation = self._accumulation

        if isinstance(data, list):
            assert type(data[0]) == Tensor, 'the type of data has to be list of torch.Tensor or torch.Tensor'
        else:
            assert type(data) == Tensor, 'the type of data has to be list of torch.Tensor or torch.Tensor'

        if accumulation is not None:
            assert type(data) == type(accumulation), \
                'the type of data ({}) is different from ' \
                'the type of the accumulation ({})'.format(
                type(data), type(accumulation))

        if isinstance(data, list):
            if accumulation is None:
                self._accumulation = [d.mul(scale) for d in data]
            else:
                self._accumulation = [acc.add(scale, d)
                                      for acc, d in zip(accumulation, data)]
        else:
            if accumulation is None:
                self._accumulation = data.mul(scale)
            else:
                self._accumulation = accumulation.add(scale, data)

    def get(self, clear=True):
        if self._accumulation is None:
            return None

        if isinstance(self._accumulation, list):
            data = [d.clone().detach() for d in self._accumulation]
        else:
            data = self._accumulation.clone().detach()

        if clear:
            self.clear()

        return data

    def clear(self):
        self._accumulation = None


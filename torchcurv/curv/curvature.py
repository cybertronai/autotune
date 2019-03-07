import torch
import torchcurv

PI_TYPE_TRACENORM = 'tracenorm'


class Curvature(object):

    def __init__(self, module, ema_decay=1., damping=1e-7):
        self._module = module
        self.ema_decay = ema_decay
        self.damping = damping

        self._data = None
        self._input_data = None
        self.ema = None
        self.inv = None

        # for vi
        self.std = None

        module.register_forward_pre_hook(self.forward_preprocess)
        module.register_backward_hook(self.backward_postprocess)

    @property
    def data(self):
        return self._data

    @property
    def bias(self):
        bias = getattr(self._module, 'bias', None)
        return False if bias is None else True

    def forward_preprocess(self, module, input):
        self.update_in_forward(input[0].data)

    def backward_postprocess(self, module, grad_input, grad_output):

        def _adjust_scale(grad_output_data):
            # for adjusting grad scale along with 'reduction' in loss function
            batch_size = grad_output_data.shape[0]
            return grad_output_data.mul(batch_size)

        grad_output_data = _adjust_scale(grad_output[0].data)
        self.update_in_backward(grad_output_data)

    def update_in_forward(self, input_data):
        self._input_data = input_data.clone()

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    # modified for vi
    def update_ema(self, n=1):
        data = self.data
        ema = self.ema
        alpha = self.ema_decay
        if ema is None or alpha == 1:
            self.ema = data
        else:
            assert type(data) == type(ema)
            if isinstance(ema, torch.Tensor):
                self.ema = data.mul(alpha/n).add(1 - alpha, ema)
            elif isinstance(ema, list):
                self.ema = [d.mul(alpha/n).add(1 - alpha, e)
                            for d, e in zip(data, ema)]
            else:
                raise TypeError
        self.zero_data()

    def update_inv(self):
        ema = self.ema

        if isinstance(ema, torch.Tensor):
            self.inv = self._inv(ema)
        elif isinstance(ema, list):
            self.inv = [self._inv(e) for e in ema]
        else:
            raise TypeError

    def _inv(self, X):
        X_damp = add_value_to_diagonal(X, self.damping)

        return torchcurv.utils.inv(X_damp)

    def precgrad(self, params):
        raise NotImplementedError

    # for vi
    def zero_data(self):
        data = self.data
        if isinstance(data, torch.Tensor):
            data.fill_(0)
        elif isinstance(data, list):
            for d in data:
                d.fill_(0)

        else:
            raise TypeError


class DiagCurvature(Curvature):

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    def _inv(self, X):
        X_damp = X.add(X.new_ones(X.shape).mul(self.damping))

        return 1 / X_damp

    def precgrad(self, params):
        precgrad = []

        for param_i, inv_i in zip(params, self.inv):
            grad = param_i.grad
            precgrad.append(inv_i.mul(grad))

        return precgrad


class KronCurvature(Curvature):

    def __init__(self, module, pi_type=PI_TYPE_TRACENORM, **kwargs):
        self.pi_type = pi_type
        self._A = None
        self._G = None

        super(KronCurvature, self).__init__(module, **kwargs)

    @property
    def data(self):
        return [self._A, self._G]

    def update_in_forward(self, input_data):
        raise NotImplementedError

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    def update_inv(self):
        A, G = self.ema

        if self.pi_type == PI_TYPE_TRACENORM:
            pi = torch.sqrt((A.trace()/A.shape[0])/(G.trace()/G.shape[0]))
        else:
            pi = 1.

        r = self.damping**0.5
        self.inv = [torchcurv.utils.inv(add_value_to_diagonal(X, value))
                    for X, value in zip([A, G], [r*pi, r/pi])]

    def precgrad(self, params):
        raise NotImplementedError

    # for vi
    def update_std(self):
        A_inv, G_inv = self.inv

        self.std = [torchcurv.utils.cholesky(X)
                    for X in [A_inv, G_inv]]


def add_value_to_diagonal(X, value):
    indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)

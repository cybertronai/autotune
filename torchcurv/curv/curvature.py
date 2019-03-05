import torch
import torchcurv

PI_TYPE_TRACENORM = 'tracenorm'


class Curvature(object):

    def __init__(self, module, ema_decay=1., damping=1e-7):
        self._module = module
        self.ema_decay = ema_decay
        self.damping = damping

        self._data = None
        self.ema = None
        self.inv = None

        module.register_backward_hook(self.backward_postprocess)

    @property
    def data(self):
        return self._data

    @property
    def bias(self):
        bias = getattr(self._module, 'bias', None)
        return False if bias is None else True

    def backward_postprocess(self, module, grad_input, grad_output):
        self.update(grad_input[0].data, grad_output[0].data)
        self.adjust_scale(grad_output[0].data)

    def update(self, input_data, grad_output_data):
        raise NotImplementedError

    def adjust_scale(self, grad_output_data):
        # for adjusting grad scale along with 'reduction' in loss function
        batch_size = grad_output_data.shape[0]
        scale = batch_size
        self._adjust_scale(scale)

    def _adjust_scale(self, scale):
        self._data.mul_(scale**2)

    def update_ema(self):
        data = self.data
        ema = self.ema
        alpha = self.ema_decay
        if ema is None or alpha == 1:
            self.ema = data
        else:
            assert type(data) == type(ema)
            if isinstance(ema, torch.Tensor):
                self.ema = data.mul(alpha).add(1 - alpha, ema)
            elif isinstance(ema, list):
                self.ema = [d.mul(alpha).add(1 - alpha, e)
                            for d, e in zip(data, ema)]
            else:
                raise TypeError

    def update_inv(self):
        ema = self.ema
        damping = self.damping

        def compute_inv(X):
            X_damp = add_value_to_diagonal(X, damping)
            return torchcurv.utils.inv(X_damp)

        if isinstance(ema, torch.Tensor):
            self.inv = compute_inv(ema)
        elif isinstance(ema, list):
            self.inv = [compute_inv(e) for e in ema]

    def precgrad(self, params):
        raise NotImplementedError


class DiagCurvature(Curvature):

    def update(self, input_data, grad_output_data):
        raise NotImplementedError

    def update_inv(self):
        ema = self.ema
        damping = self.damping
        ema_damp = ema.add(ema.new_ones(ema.shape[0])).mul(damping)
        self.inv = 1 / ema_damp

    def precgrad(self, params):
        raise NotImplementedError


class KronCurvature(Curvature):

    def __init__(self, module, pi_type=PI_TYPE_TRACENORM, **kwargs):
        self.pi_type = pi_type
        self._A = None
        self._G = None

        module.register_forward_pre_hook(self.forward_preprocess)
        super(KronCurvature, self).__init__(module, **kwargs)

    @property
    def data(self):
        return [self._A, self._G]

    def forward_preprocess(self, module, input):
        self.update_A(input[0].data)

    def backward_postprocess(self, module, grad_input, grad_output):
        self.update_G(grad_output[0].data)
        self.adjust_scale(grad_output[0].data)

    def update(self, input_data, grad_output_data):
        # KronCurvature class doesn't update data directly
        pass

    def update_A(self, input_data):
        raise NotImplementedError

    def update_G(self, grad_output_data):
        raise NotImplementedError

    def _adjust_scale(self, scale):
        self._G.mul_(scale**2)

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


def add_value_to_diagonal(X, value):
    indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)


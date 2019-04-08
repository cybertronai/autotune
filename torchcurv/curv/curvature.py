import torch
import torchcurv

PI_TYPE_TRACENORM = 'tracenorm'


class Curvature(object):

    def __init__(self, module, ema_decay=1., damping=1e-7, post_curv=None):
        self._module = module
        self.ema_decay = ema_decay
        self._damping = damping
        self.l2_reg = 0

        self._data = None
        self._acc_data = None
        self.ema = None
        self.inv = None
        self.std = None

        self.post_curv = post_curv

        module.register_forward_hook(self.forward_postprocess)
        module.register_backward_hook(self.backward_postprocess)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def module(self):
        return self._module

    @property
    def bias(self):
        bias = getattr(self._module, 'bias', None)
        return False if bias is None else True

    @property
    def damping(self):
        return self._damping + self.l2_reg

    def forward_postprocess(self, module, input, output):
        assert self._module == module

        data_input = input[0].detach()

        setattr(self._module, 'data_input', data_input)
        setattr(self._module, 'data_output', output)

        self.update_in_forward(data_input)

    def backward_postprocess(self, module, grad_input, grad_output):
        assert self._module == module

        index = 1 if self.bias else 0
        grad_input = grad_input[index].detach()
        grad_output = grad_output[0]

        setattr(module, 'grad_input', grad_input)
        setattr(module, 'grad_output', grad_output)

        self.update_in_backward(grad_output)

        # adjust grad scale along with 'reduction' in loss function
        batch_size = grad_output.shape[0]
        self.adjust_data_scale(batch_size**2)

    def adjust_data_scale(self, scale):
        self._data = [d.mul(scale) for d in self._data]

    def update_in_forward(self, data_input):
        pass

    def update_in_backward(self, grad_output):
        raise NotImplementedError

    def step(self, update_std=False):
        # TODO(oosawak): Add check for ema/inv timing
        self.update_ema()
        self.update_inv()
        if update_std:
            self.update_std()

    def update_ema(self):
        data = self.data
        ema = self.ema
        alpha = self.ema_decay
        if ema is None or alpha == 1:
            self.ema = [d.clone() for d in data]
        else:
            self.ema = [d.mul(alpha).add(1 - alpha, e)
                        for d, e in zip(data, ema)]

    def update_inv(self):
        ema = self.ema
        self.inv = [self._inv(e) for e in ema]

    def _inv(self, X):
        X_damp = add_value_to_diagonal(X, self.damping)

        return torchcurv.utils.inv(X_damp)

    def precondition_grad(self, params):
        raise NotImplementedError

    def update_std(self):
        raise NotImplementedError

    def sample_params(self, params, mean, std_scale):
        raise NotImplementedError


class DiagCurvature(Curvature):

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    def _inv(self, X):
        X_damp = X.add(X.new_ones(X.shape).mul(self.damping))

        return 1 / X_damp

    def precondition_grad(self, params):
        for p, inv in zip(params, self.inv):
            preconditioned_grad = inv.mul(p.grad)
            p.grad.copy_(preconditioned_grad)

    def update_std(self):
        self.std = [inv.sqrt() for inv in self.inv]

    def sample_params(self, params, mean, std_scale):
        for p, m, std in zip(params, mean, self.std):
            noise = torch.randn_like(m)
            p.data.copy_(torch.addcmul(m, std_scale, noise, std))


class KronCurvature(Curvature):

    def __init__(self, module, ema_decay=1., damping=1e-7,
                 post_curv=None, pi_type=PI_TYPE_TRACENORM):
        self.pi_type = pi_type
        self._A = None
        self._G = None

        super(KronCurvature, self).__init__(module, ema_decay=ema_decay, damping=damping,
                                            post_curv=post_curv)

    @property
    def data(self):
        return [self._A, self._G]

    @data.setter
    def data(self, value):
        self._A, self._G = value

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    def update_in_forward(self, input_data):
        raise NotImplementedError

    def update_in_backward(self, grad_output_data):
        raise NotImplementedError

    def adjust_data_scale(self, scale):
        self._G.mul_(scale)

    def update_inv(self):
        A, G = self.ema

        if self.pi_type == PI_TYPE_TRACENORM:
            pi = torch.sqrt((A.trace()/A.shape[0])/(G.trace()/G.shape[0]))
        else:
            pi = 1.

        r = self.damping**0.5
        self.inv = [torchcurv.utils.inv(add_value_to_diagonal(X, value))
                    for X, value in zip([A, G], [r*pi, r/pi])]

    def precondition_grad(self, params):
        raise NotImplementedError

    def update_std(self):
        A_inv, G_inv = self.inv

        self.std = [torchcurv.utils.cholesky(X)
                    for X in [A_inv, G_inv]]

    def sample_params(self, params, mean, std_scale):
        raise NotImplementedError


def add_value_to_diagonal(X, value):
    indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)

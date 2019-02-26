import torch


class Curvature(object):

    def __init__(self):
        pass


class DiagCurvature(Curvature):

    def __init__(self):
        pass


def update_input(self, input, output):
    self.input_data = input[0].data


def update_grad_output(self, grad_input, grad_output):
    self.grad_output_data = grad_output[0].data


class KronCurvature(Curvature):

    def __init__(self,
                 module,
                 cov_ema_decay,
                 damping,
                 pi_type):
        self._module = module
        self.bias = False if module.bias is None else True
        self.cov_ema_decay = cov_ema_decay
        self.damping = damping
        self.A = None
        self.G = None
        self.A_ema = None
        self.G_ema = None
        self.pi_type = pi_type
        module.register_forward_hook(update_input)
        module.register_backward_hook(update_grad_output)

    def compute_A(self):
        raise NotImplementedError

    def compute_G(self):
        raise NotImplementedError

    def update_covs_ema(self):
        A, G = self.A, self.G
        if self.A_ema is None and self.G_ema is None:
            self.A_ema, self.G_ema = A, G
        else:
            alpha = self.cov_ema_decay
            A_ema = A.mul(alpha).add(1-alpha, self.A_ema)
            G_ema = G.mul(alpha).add(1-alpha, self.G_ema)
            self.A_ema, self.G_ema = A_ema, G_ema

    def compute_pi_tracenorm(self, A, G):
        A_size, G_size = A.shape[0], G.shape[0]

        return torch.sqrt((A.trace()/(A_size))/(G.trace()/(G_size)))

    def compute_damped_covs(self, A, G):
        if self.pi_type == 'trace_norm':
            pi = self.compute_pi_tracenorm(A, G)
        else:
            pi = 1
        r = self.damping**0.5
        pi = float(pi)
        A_damping = torch.diag(torch.ones(A.shape[0], device=A.device))
        G_damping = torch.diag(torch.ones(G.shape[0], device=G.device))
        A.add_(r*pi, A_damping)
        G.add_(r/pi, G_damping)

        return A, G

    def compute_precgrad(self, params):
        raise NotImplementedError

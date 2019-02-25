import torch


class Curvature(object):

    def __init__(self):
        pass


class DiagCurvature(Curvature):

    def __init__(self):
        pass


def update_input(self, input, output):
    self.input = input[0].data


def update_grad_output(self, grad_input, grad_output):
    self.grad_output = grad_output[0].data


class KronCurvature(Curvature):

    def __init__(self,
                 module,
                 cov_ema_decay=0.99,
                 damping=0.01,
                 pi_type='trace_norm'
                 ):
        self._module = module
        self.bias = False if module.bias is None else True
        self.cov_ema_decay = cov_ema_decay
        self.damping = damping
        self.covs = None
        self.covs_ema = None
        self.pi_type = pi_type
        module.register_forward_hook(update_input)
        module.register_backward_hook(update_grad_output)

    def compute_A(self):
        raise NotImplementedError

    def compute_G(self):
        raise NotImplementedError

    def update_covs_ema(self):
        covs = self.covs
        if self.covs_ema is None:
            self.covs_ema = covs
        else:
            alpha = self.cov_ema_decay
            A, G = covs
            A_ema = A.mul(alpha).add(1-alpha, self.covs_ema[0])
            G_ema = G.mul(alpha).add(1-alpha, self.covs_ema[1])
            self.covs_ema = A_ema, G_ema

    def compute_pi_tracenorm(self, covs):
        A, G = covs
        A_size, G_size = A.shape[0], G.shape[0]

        return torch.sqrt((A.trace()/(A_size))/(G.trace()/(G_size)))

    def compute_damped_covs(self, covs):
        if self.pi_type == 'trace_norm':
            pi = self.compute_pi_tracenorm(covs)
        else:
            pi = 1
        r = self.damping**0.5
        pi = float(pi)
        A, G = covs
        A_damping = torch.diag(torch.ones(A.shape[0], device=A.device))
        G_damping = torch.diag(torch.ones(G.shape[0], device=G.device))
        A.add_(r*pi, A_damping)
        G.add_(r/pi, G_damping)

        return A, G

    def compute_precgrad(self, params):
        raise NotImplementedError

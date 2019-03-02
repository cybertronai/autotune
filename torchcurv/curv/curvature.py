import torch


class Curvature(object):

    def __init__(self):
        pass


class DiagCurvature(Curvature):

    def __init__(self):
        pass


class KronCurvature(Curvature):

    def update_input(self, module, input):
        self.compute_A(input[0].data)

    def update_grad_output(self, module, grad_input, grad_output):
        self.compute_G(grad_output[0].data)

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

        module.register_forward_pre_hook(self.update_input)
        module.register_backward_hook(self.update_grad_output)

    def compute_A(self, input_data):
        raise NotImplementedError

    def compute_G(self, grad_output_data):
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
        A_indices = torch.LongTensor([[i, i] for i in range(A.shape[0])])
        G_indices = torch.LongTensor([[i, i] for i in range(G.shape[0])])
        A.index_put_(tuple(A_indices.t()), A.diagonal().add(torch.ones(
            A_indices.shape[0], device=A.device).mul(r*pi)))
        G.index_put_(tuple(G_indices.t()), G.diagonal().add(torch.ones(
            G_indices.shape[0], device=G.device).mul(r/pi)))

        return A, G

    def compute_precgrad(self, params):
        raise NotImplementedError


class KronCurvatureConnection(KronCurvature):

    def __init__(self,
                 module,
                 cov_ema_decay,
                 damping,
                 pi_type):
        super(KronCurvatureConnection, self).__init__(module,
                                                      cov_ema_decay,
                                                      damping,
                                                      pi_type)
        self.bias = False if module.bias is None else True

    def compute_A(self, input_data):
        raise NotImplementedError

    def compute_G(self, grad_output_data):
        raise NotImplementedError

    def compute_precgrad(self, params):
        raise NotImplementedError

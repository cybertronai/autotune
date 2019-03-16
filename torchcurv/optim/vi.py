from torchcurv.optim import SecondOrderOptimizer
from torchcurv.utils import TensorAccumulator


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, curv_type, lr=0.01, momentum=0.9, momentum_type='precgrad', adjust_momentum=False, l2_reg=0, weight_decay=0, num_samples=10, std_scale=4e-6, **curv_kwargs):
        super(VIOptimizer, self).__init__(model, curv_type, lr=lr, momentum=momentum,
                                          momentum_type=momentum_type, adjust_momentum=adjust_momentum, l2_reg=l2_reg, weight_decay=weight_decay, **curv_kwargs)
        self.defaults['num_samples'] = num_samples
        self.defaults['std_scale'] = std_scale

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        def closure():
            # forward/backward
            return loss, output
        """

        n = self.defaults['num_samples']
        std_scale = self.defaults['std_scale']
        acc_loss = TensorAccumulator()
        acc_ouput = TensorAccumulator()

        for i in range(n):

            # sampling
            for group in self.param_groups:
                params, mean, curv = group['params'], group['mean'], group['curv']
                curv.sample_params(params, mean, std_scale)

            # forward and backward
            loss, output = closure()

            acc_loss.update(loss, scale=1/n)
            acc_ouput.update(output, scale=1/n)

            # update buf
            for group in self.param_groups:
                params = group['params']
                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/n)
                    grads = [p.grad for p in params]
                    group['acc_grads'].update(grads, scale=1/n)

        # update distribution
        for group in self.param_groups:
            params = group['params']

            acc_grads = group['acc_grads'].get()
            for p, acc_grad in zip(params, acc_grads):
                p.grad.data.copy_(acc_grad)

            self.update_preprocess(group)

            # update covariance
            curv = group['curv']
            if curv is not None:
                curv.data = group['acc_curv'].get()
                curv.update_ema()
                curv.update_inv()
                curv.update_std()
                curv.precondition_grad(params)

            # update mean
            self.update(group, target='mean')

        loss, output = acc_loss.get(), acc_ouput.get()

        return loss, output


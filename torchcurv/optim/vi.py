import torch
from torchcurv.optim import SecondOrderOptimizer
from torchcurv.utils import TensorAccumulator


class VIOptimizer(SecondOrderOptimizer):

    def __init__(self, model, curv_type, lr=0.01, momentum=0.9, momentum_type='precgrad', adjust_momentum=False, l2_reg=0, weight_decay=0, num_samples=10, std_scale=4e-6, **curv_kwargs):
        super(VIOptimizer, self).__init__(model, curv_type, lr=lr, momentum=momentum,
                                          momentum_type=momentum_type, adjust_momentum=adjust_momentum, l2_reg=l2_reg, weight_decay=weight_decay, **curv_kwargs)
        self.defaults['num_samples'] = num_samples
        self.defaults['std_scale'] = std_scale

        self.state['step'] = 0

        for group in self.param_groups:
            group['mean'] = [p.clone().detach() for p in group['params']]
            for m in group['mean']:
                state = self.state[m]
                state['momentum_buffer'] = torch.zeros_like(m.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        def closure():
            # forward/backward
            return loss, output
        """

        n = self.defaults['num_samples'] if self.state['step'] > 0 else 1
        std_scale = self.defaults['std_scale']
        acc_loss = TensorAccumulator()
        acc_output = TensorAccumulator()

        for i in range(n):

            # sampling
            for group in self.param_groups:
                curv = group['curv']
                if curv is not None:
                    params, mean = group['params'], group['mean']
                    curv.sample_params(params, mean, std_scale)

            # forward and backward
            loss, output = closure()

            acc_loss.update(loss, scale=1/n)
            acc_output.update(output, scale=1/n)

            # update buf
            for group in self.param_groups:
                params = group['params']

                self.backward_postprocess(group)

                grads = [p.grad.data for p in params]
                group['acc_grads'].update(grads, scale=1/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/n)

        self.state['step'] += 1

        # update distribution
        for group in self.param_groups:

            acc_grads = group['acc_grads'].get()

            curv = group['curv']
            if curv is not None:
                mean = group['mean']

                # save accumulated grad
                for m, acc_grad in zip(mean, acc_grads):
                    m.grad = acc_grad.clone()

                # update covariance
                curv.data = group['acc_curv'].get()
                curv.update_ema()
                curv.update_inv()
                curv.update_std()
                curv.precondition_grad(mean)

                # update mean
                self.update(group, target='mean')
            else:
                params = group['params']

                # save accumulated grad
                for p, acc_grad in zip(params, acc_grads):
                    p.grad = acc_grad.clone()

                # update params
                self.update(group)

        loss, output = acc_loss.get(), acc_output.get()

        return loss, output


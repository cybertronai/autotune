from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class DistributedFirstOrderOptimizer(Optimizer):

    def __init__(self, optimizer, model, dist):
        super(DistributedFirstOrderOptimizer, self).__setattr__(
           'actual_optimizer', optimizer
        )
        super(DistributedFirstOrderOptimizer, self).__setattr__(
            'model', model
        )
        super(DistributedFirstOrderOptimizer, self).__setattr__(
            'dist', dist
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            world_size = self.dist.get_world_size()
            grads = [p.grad for p in self.model.parameters()]
            # pack
            packed_tensor = parameters_to_vector(grads)
            # all reduce
            self.dist.all_reduce(packed_tensor)
            # unpack
            vector_to_parameters(packed_tensor.div_(world_size), grads)

        self.actual_optimizer.step(closure=None)

        return loss

    def __getattr__(self, item):
        return getattr(self.actual_optimizer, item)

    def __setattr__(self, key, value):
        setattr(self.actual_optimizer, key, value)


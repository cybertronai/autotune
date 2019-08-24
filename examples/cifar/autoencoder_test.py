import os
import sys
from typing import List

import numpy as np
# import torch
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from PIL import Image
from torch import optim

# Test exact Hessian computation

module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path)
import util as u


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TinyMNIST(datasets.MNIST):
    """Dataset for autoencoder task."""

    # 60k,1,new_dim,new_dim
    def __init__(self, root, data_width=4, targets_width=4, dataset_size=60000, download=True):
        super().__init__(root, download)

        # Put both data and targets on GPU in advance
        self.data = self.data[:dataset_size, :, :]
        new_data = np.zeros((self.data.shape[0], data_width, data_width))
        new_targets = np.zeros((self.data.shape[0], targets_width, targets_width))
        for i in range(self.data.shape[0]):
            arr = self.data[i, :].numpy().astype(np.uint8)
            im = Image.fromarray(arr)
            im.thumbnail((data_width, data_width), Image.ANTIALIAS)
            new_data[i, :, :] = np.array(im) / 255
            im = Image.fromarray(arr)
            im.thumbnail((targets_width, targets_width), Image.ANTIALIAS)
            new_targets[i, :, :] = np.array(im) / 255

        self.data = torch.from_numpy(new_data).float()
        self.data = self.data.unsqueeze(1)
        self.targets = torch.from_numpy(new_targets).float()
        self.targets = self.targets.unsqueeze(1)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Net(nn.Module):
    def __init__(self, d: List[int], nonlin=False):
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=False)
            setattr(linear, 'name', f'{i:02d}-linear')
            self.layers.append(linear)
            self.all_layers.append(linear)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)


def autoencoder_minimize_test():
    """Minimize autoencoder for a few steps."""
    data_width = 4
    targets_width = 2
    batch_size = 64
    dataset = TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d = data_width ** 2
    model = Net([d, targets_width**2])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    loss = 0
    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        if i == 0:
            assert loss > 0.03
        print(loss)
        loss.backward()
        optimizer.step()

    assert loss < 0.015


def autoencoder2_minimize_test():
    """deeper autoencoder."""
    data_width = 4
    targets_width = 2

    batch_size = 64
    dataset = TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    model = Net([d1, d2, d3], nonlin=True)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    loss = 0
    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        if i == 0:
            assert loss > 0.055
            pass
        print(loss)
        loss.backward()
        optimizer.step()

    assert loss < 0.032


def autoencoder_newton_test():
    """Newton method towards minimum"""

    image_size = 3
    batch_size = 64
    dataset = TinyMNIST('/tmp', download=True, data_width=image_size, targets_width=image_size, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d = image_size ** 2   # hidden layer size
    u.seed_random(1)
    model = Net([d, d])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    def unvec(x):
        return u.unvec(x, d)

    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        print(loss)
        if i > 0:
            assert loss < 1e-9

        loss.backward()
        W = model.layers[0].weight
        grad = u.vec(W.grad)

        loss = loss_fn(model(data), targets)
        H = u.hessian(loss, W)
        H = H.transpose(0, 1).transpose(2, 3).reshape(d**2, d**2)

        W1 = unvec(u.vec(W) - u.pinv(H) @ grad)
        W.data.copy_(W1)


def autoencoder_newton_transposed_test():
    """Newton method towards minimum, without transposing Hessian"""

    image_size = 3
    batch_size = 64
    dataset = TinyMNIST('/tmp', download=True, data_width=image_size, targets_width=image_size, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d = image_size ** 2   # hidden layer size
    u.seed_random(1)
    model = Net([d, d])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    def untvec(x):
        return u.untvec(x, d)

    for i in range(10):
        data, targets = next(iter(trainloader))
        optimizer.zero_grad()
        loss = loss_fn(model(data), targets)
        print(loss)
        if i > 0:
            assert loss < 1e-9

        loss.backward()
        W = model.layers[0].weight
        grad = u.tvec(W.grad)

        loss = loss_fn(model(data), targets)
        H = u.hessian(loss, W)
        H = H.reshape(d**2, d**2)

        W1 = untvec(u.tvec(W) - grad @ u.pinv(H))
        W.data.copy_(W1)


def manual_linear_hessian_test():
    u.seed_random(1)

    data_width = 3
    targets_width = 2
    batch_size = 3
    dataset = TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d1 = data_width ** 2   # hidden layer size, visible size, output size
    d2 = targets_width ** 2   # hidden layer size, visible size, output size
    n = batch_size
    model = Net([d1, d2])
    layer = model.layers[0]
    W = model.layers[0].weight

    skip_hooks = False

    def capture_activations(module, input, _output):
        if skip_hooks:
            return
        assert not hasattr(module, 'activations'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(input) == 1, "this works for single input layers only"
        setattr(module, "activations", input[0].detach())

    def capture_backprops(module: nn.Module, _input, output):
        if skip_hooks:
            return
        assert not hasattr(module, 'backprops'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(output) == 1, "this works for single variable layers only"
        setattr(module, "backprops", output[0])

    layer.register_forward_hook(capture_activations)
    layer.register_backward_hook(capture_backprops)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    #    def unvec(x): return u.unvec(x, d)

    # Gradient
    data, targets = next(iter(trainloader))
    loss = loss_fn(model(data), targets)
    loss.backward()
    A = layer.activations.t()
    assert A.shape == (d1, n)

    # add factor of n here because backprop computes loss averaged over batch, while we need per-example loss backprop
    B = layer.backprops.t() * n
    assert B.shape == (d2, n)
    u.check_close(B @ A.t() / n, W.grad)

    # Hessian
    skip_hooks = True
    loss = loss_fn(model(data), targets)
    H = u.hessian(loss, W)
    H = H.transpose(0, 1).transpose(2, 3).reshape(d1 * d2, d1 * d2)
    print(H)

    # compute B matrices
    Bs_t = []   # one matrix per class, storing backprops for current layer
    skip_hooks = False
    id_mat = torch.eye(d2)
    for out_idx in range(d2):
        u.zero_grad(model)
        output = model(data)
        _loss = loss_fn(output, targets)

        ei = id_mat[out_idx]
        bval = torch.stack([ei]*batch_size)
        output.backward(bval)
        Bs_t.append(layer.backprops)

    A_t = layer.activations

    # batch output Jacobian, each row corresponds to example,output pair
    Amat = torch.cat([A_t]*d2, dim=0)
    Bmat = torch.cat(Bs_t, dim=0)
    Jb = u.khatri_rao_t(Amat, Bmat)
    H2 = Jb.t() @ Jb / n

    u.check_close(H, H2)


def manual_nonlinear_hessian_test():
    u.seed_random(1)

    data_width = 4
    targets_width = 2

    batch_size = 5
    dataset = TinyMNIST('/tmp', download=True, data_width=data_width, targets_width=targets_width, dataset_size=batch_size)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    u.seed_random(1)
    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    n = batch_size
    model = Net([d1, d2, d3], nonlin=True)
    layer = model.layers[0]
    W = model.layers[0].weight

    skip_hooks = False

    def capture_activations(module, input, _output):
        if skip_hooks:
            return
        assert not hasattr(module, 'activations'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(input) == 1, "this works for single input layers only"
        setattr(module, "activations", input[0].detach())

    def capture_backprops(module: nn.Module, _input, output):
        if skip_hooks:
            return
        assert not hasattr(module, 'backprops'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(output) == 1, "this works for single variable layers only"
        setattr(module, "backprops", output[0])

    layer.register_forward_hook(capture_activations)
    layer.register_backward_hook(capture_backprops)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    #    def unvec(x): return u.unvec(x, d)

    # Gradient
    data, targets = next(iter(trainloader))
    loss = loss_fn(model(data), targets)
    loss.backward()
    A = layer.activations.t()
    assert A.shape == (d1, n)

    # add factor of n here because backprop computes loss averaged over batch, while we need per-example loss backprop
    B = layer.backprops.t() * n
    assert B.shape == (d2, n)
    u.check_close(B @ A.t() / n, W.grad)

    # Hessian
    skip_hooks = True
    loss = loss_fn(model(data), targets)
    H_autograd = u.hessian(loss, W)
    H_autograd_t = H_autograd.transpose(0, 1).transpose(2, 3).reshape(d1 * d2, d1 * d2)
    print(H_autograd_t)

    # compute B matrices
    Bs_t = []   # one matrix per class, storing backprops for current layer
    skip_hooks = False
    id_mat = torch.eye(o)
    for out_idx in range(o):
        u.zero_grad(model)
        output = model(data)
        _loss = loss_fn(output, targets)

        ei = id_mat[out_idx]
        bval = torch.stack([ei]*batch_size)
        output.backward(bval)
        Bs_t.append(layer.backprops)

    A_t = layer.activations

    # batch output Jacobian, each row corresponds to example,output pair
    Amat = torch.cat([A_t]*o, dim=0)
    Bmat = torch.cat(Bs_t, dim=0)
    assert Amat.shape == (n*o, d1)
    assert Bmat.shape == (n*o, d2)
    Jb = u.khatri_rao_t(Amat, Bmat)
    H_manual = Jb.t() @ Jb / n
    u.check_close(H_manual, H_autograd_t)

    # we can recover the same row-vectorized order as PyTorch autograd by swapping order in Khatri-Rao
    Jb2 = u.khatri_rao_t(Bmat, Amat)
    H_manual2 = Jb2.t() @ Jb2 / n
    u.check_close(H_manual2, H_autograd.reshape(d1 * d2, d1 * d2))


if __name__ == '__main__':
    #    autoencoder_minimize_test()
    #    autoencoder2_minimize_test()
    #    autoencoder_newton_test()
    #    autoencoder_newton_transposed_test()
    #    manual_linear_hessian_test()
    manual_nonlinear_hessian_test()

    #    u.run_all_tests(sys.modules[__name__])

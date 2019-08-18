# Take simple example, plot per-layer stats over time
import argparse
import json
import os
import shutil
import sys
from importlib import import_module


from typing import Optional, Tuple, Callable

import scipy
import torch
import torch.nn as nn
from torchcurv.optim import SecondOrderOptimizer

import numpy as np
import torch.nn.functional as F
try:
    import wandb
except Exception as e:
    print(f"wandb crash with {e}")


from torchvision import datasets, transforms, models
import torch

import torchcurv
from torchcurv.optim import SecondOrderOptimizer, VIOptimizer
from torchcurv.utils import Logger

DATASET_MNIST = 'MNIST'
IMAGE_SIZE = 28

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def install_pdb_handler():
    """Automatically start pdb:
      1. CTRL+\\ breaks into pdb.
      2. pdb gets launched on exception.
  """

    import signal
    import pdb

    def handler(_signum, _frame):
        pdb.set_trace()

    signal.signal(signal.SIGQUIT, handler)

    # Drop into PDB on exception
    # from https://stackoverflow.com/questions/13174412
    def info(type_, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type_, value, tb)
        else:
            import traceback
            import pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type_, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.
            pdb.pm()

    sys.excepthook = info


install_pdb_handler()


class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
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


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_MNIST], default=DATASET_MNIST,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
    parser.add_argument('--random_crop', action='store_true',
                        help='[data augmentation] random crop')
    parser.add_argument('--random_horizontal_flip', action='store_true',
                        help='[data augmentation] random horizontal flip')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default=None,
                        help='name of file which defines the architecture')
    parser.add_argument('--arch_name', type=str, default='LeNet5',
                        help='name of the architecture')
    parser.add_argument('--arch_args', type=json.loads, default=None,
                        help='[JSON] arguments for the architecture')
    parser.add_argument('--optim_name', type=str, default=SecondOrderOptimizer.__name__,
                        help='name of the optimizer')
    parser.add_argument('--optim_args', type=json.loads, default=None,
                        help='[JSON] arguments for the optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=None,
                        help='[JSON] arguments for the curvature')
    # Options
    parser.add_argument('--download', action='store_true', default=False,
                        help='if True, downloads the dataset (CIFAR-10 or 100) from the internet')
    parser.add_argument('--create_graph', action='store_true', default=False,
                        help='create graph of the derivative')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path for resume training')
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save output files')
    parser.add_argument('--config', default='configs/cifar10/mlp_autoencoder.json',
                        help='config file path')
    parser.add_argument('--fisher_mc_approx', action='store_true', default=False,
                        help='if True, Fisher is estimated by MC sampling')
    parser.add_argument('--fisher_num_mc', type=int, default=1,
                        help='number of MC samples for estimating Fisher')

    args = parser.parse_args()

    run_name = args.config
    run_name = os.path.basename(run_name)
    run_name = run_name.rsplit('.', 1)[0]  # extract filename without .json suffix

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        # wandb.init(project='pytorch-curv', name=run_name)
        wandb.config['config'] = args.config
        wandb.config['batch'] = args.batch_size
        wandb.config['optim'] = args.optim_name
    except Exception as e:
        # print(f"wandb crash with {e}")
        pass

    dict_args = vars(args)

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup data augmentation & data pre processing
    train_transforms, val_transforms = [], []
    if args.random_crop:
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if args.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())

    print(args.batch_size)

    if args.normalizing_data:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transforms.append(normalize)
        val_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)

    assert args.dataset == DATASET_MNIST
    num_classes = 10
    dataset_class = FastMNIST

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d = [IMAGE_SIZE**2, 8, 8, 8, 8, 1]
            self.d = d
            self.W0 = nn.Linear(d[0], d[1], bias=False)
            self.W1 = nn.Linear(d[1], d[2], bias=False)
            self.W2 = nn.Linear(d[2], d[3], bias=False)
            self.W3 = nn.Linear(d[3], d[4], bias=False)
            self.W4 = nn.Linear(d[4], d[5], bias=False)

        def forward(self, X1: torch.Tensor):
            result = X1.reshape((-1, self.d[0]))
            result = F.relu(self.W0(result))
            result = F.relu(self.W1(result))
            result = F.relu(self.W2(result))
            result = F.relu(self.W3(result))
            result = self.W4(result)
            return result

    train_dataset = dataset_class(
        root=args.root, train=True, download=args.download, transform=train_transform)
    val_dataset = dataset_class(
        root=args.root, train=False, download=args.download, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    model = Net()

    arch_kwargs = {} if args.arch_args is None else args.arch_args
    arch_kwargs['num_classes'] = num_classes

    setattr(model, 'num_classes', num_classes)
    model = model.to(device)

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0.01, momentum=0.9, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", update_inv=False, precondition_grad=False)
    curv_args = dict(damping=0, ema_decay=1)  # todo: damping
    optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)

    start_epoch = 1

    # Copy this file & config to args.out
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    shutil.copy(os.path.realpath(__file__), args.out)

    if args.config is not None:
        shutil.copy(args.config, args.out)
    if args.arch_file is not None:
        shutil.copy(args.arch_file, args.out)

    # Setup logger
    logger = Logger(args.out, args.log_file_name)
    logger.start()

    # Run training
    for epoch in range(start_epoch, args.epochs + 1):

        # train
        accuracy, loss, confidence = train(model, device, train_loader, optimizer, epoch, args, logger)

        print('Printing maximum learning rate/batch-size for layer 3')
        n = args.batch_size
        At = model.W3.data_input
        A = At.t()

        # matrix of backprops, add factor n to remove dependence on batch-size
        Bt = model.W3.grad_output * n

        # gradients, n,d
        G = khatri_rao_t(At, Bt)
        assert False, "Implement Hessian here"

        # save log
        iteration = epoch * len(train_loader)
        log = {'epoch': epoch, 'iteration': iteration,
               'accuracy': accuracy, 'loss': loss, 'confidence': confidence,
               'val_accuracy': 0, 'val_loss': 0,
               'lr': optimizer.param_groups[0]['lr'],
               'momentum': optimizer.param_groups[0].get('momentum', 0)}
        logger.write(log)

        # save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            path = os.path.join(args.out, 'epoch{}.ckpt'.format(epoch))
            data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(data, path)


def train(model, device, train_loader, optimizer, epoch, args, logger):
    model.train()

    loss = None
    confidence = {'top1': 0, 'top1_true': 0, 'top1_false': 0, 'true': 0, 'false': 0}
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch

    last_elapsed_time = 0
    interval_ms = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        for name, param in model.named_parameters():
            attr = 'p_pre_{}'.format(name)
            setattr(model, attr, param.detach().clone())

        # update params
        def closure():
            optimizer.zero_grad()
            output = model(data)

            err = output - target.float()
            loss = torch.sum(err * err) / 2 / len(data) / IMAGE_SIZE**2
            loss.backward(create_graph=args.create_graph)

            return loss, output

        loss, output = optimizer.step(closure=closure)
        loss = loss.item()

        iteration = base_num_iter + batch_idx + 1
        total_data_size += len(data)

        if batch_idx % args.log_interval == 0:
            elapsed_time = logger.elapsed_time
            if last_elapsed_time:
                interval_ms = 1000*(elapsed_time - last_elapsed_time)
            last_elapsed_time = elapsed_time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, '
                  'Accuracy: {:.0f}/{} ({:.2f}%), '
                  'Elapsed Time: {:.1f}s'.format(
                epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch,
                loss, 0, total_data_size, 0, elapsed_time))

            # save log
            lr = optimizer.param_groups[0]['lr']
            log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                   'accuracy': 0, 'loss': loss, 'lr': lr, 'step_ms': interval_ms}
            try:
                wandb.log(log, step=iteration*args.batch_size)
            except Exception as e:
                print(f"wandb crash with {e}")

            for name, param in model.named_parameters():
                attr = 'p_pre_{}'.format(name)
                p_pre = getattr(model, attr)
                p_norm = param.norm().item()
                p_shape = list(param.size())
                p_pre_norm = p_pre.norm().item()
                g_norm = param.grad.norm().item()
                upd_norm = param.sub(p_pre).norm().item()
                noise_scale = getattr(param, 'noise_scale', 0)

                p_log = {'p_shape': p_shape, 'p_norm': p_norm, 'p_pre_norm': p_pre_norm,
                         'g_norm': g_norm, 'upd_norm': upd_norm, 'noise_scale': noise_scale}
                log[name] = p_log

            logger.write(log)

    return 0, loss, confidence


def validate(model, device, val_loader, optimizer):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:

            data, target = data.to(device), target.to(device)

            if isinstance(optimizer, VIOptimizer):
                output = optimizer.prediction(data)
            else:
                output = model(data)

            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    print('\nEval: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), val_accuracy))

    return val_accuracy, val_loss


### utility functions
def vec(a):
    """vec operator, stack columns of the matrix into single column matrix."""
    assert len(a.shape) == 2
    return a.t().reshape(-1, 1)


def unvec(a, rows):
    """reverse of vec, rows specifies number of rows in the final matrix."""
    assert len(a.shape) == 2
    assert a.shape[0] % rows == 0
    cols = a.shape[0] // rows
    return a.reshape(cols, -1).t()


def kron(a, b):
    return torch.einsum("ab,cd->acbd", a, b).view(a.size(0) * b.size(0), a.size(1) * b.size(1))


def l2_norm(mat):
    return max(torch.eig(mat).eigenvalues.flatten())


def inv_square_root(mat):
    assert type(mat) == np.ndarray
    return scipy.linalg.inv(scipy.linalg.sqrtm(mat))


def pinv_square_root(mat):
    assert type(mat) == np.ndarray
    return scipy.linalg.inv(scipy.linalg.sqrtm(mat))


def rank(mat):
    """Effective rank of matrix."""
    return torch.trace(mat) / l2_norm(mat)


def outer(x, y):
    return x.unsqueeze(1) @ y.unsqueeze(0)


def toscalar(x):
    if hasattr(x, 'item'):
        return x.item()
    x = to_numpy(x).flatten()
    assert len(x) == 1
    return x[0]


def to_numpy(x, dtype=np.float32):
    """Utility function to convert object to numpy array."""
    if hasattr(x, 'numpy'):  # PyTorch tensor
        return x.detach().numpy().astype(dtype)
    elif type(x) == np.ndarray:
        return x.astype(dtype)
    else:  # Some Python type
        return np.array(x).astype(dtype)


def khatri_rao(A, B):
    """Khatri-Rao product, see
    Section 2.6 of Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM review 51.3 (2009): 455-500"""
    assert A.shape[1] == B.shape[1]
    return torch.einsum("ik,jk->ijk", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])


# Autograd functions, from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def test_khatri_rao():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[5, 12], [7, 16], [15, 24], [21, 32]])
    check_close(khatri_rao(A, B), C)


def khatri_rao_t(A, B):
    """Like Khatri-Rao, but iterators over rows of matrices instead of cols"""
    assert A.shape[0] == B.shape[0]
    return torch.einsum("ki,kj->kij", A, B).reshape(A.shape[0], A.shape[1] * B.shape[1])


def test_khatri_rao_t():
    A = torch.tensor([[-2., -1.],
                      [0., 1.],
                      [2., 3.]])
    B = torch.tensor([[-4.],
                      [1.],
                      [6.]])
    C = torch.tensor([[8., 4.],
                      [0., 1.],
                      [12., 18.]])
    check_close(khatri_rao_t(A, B), C)


def pinv(mat, eps=1e-4):
    """Computes pseudo-inverse of mat, treating eigenvalues below eps as 0."""

    # TODO(y): make eps scale invariant by diving by norm first
    u, s, v = torch.svd(mat)
    si = torch.where(s > eps, 1/s, s)
    return u @ torch.diag(si) @ v.t()


def pinv_square_root(mat, eps=1e-4):
    u, s, v = torch.svd(mat)
    si = torch.where(s > eps, 1/torch.sqrt(s), s)
    return u @ torch.diag(si) @ v.t()


def check_close(observed, truth):
    truth = to_numpy(truth)
    observed = to_numpy(observed)
    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    np.testing.assert_allclose(truth, observed, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    main()

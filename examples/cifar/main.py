import os
import argparse
from importlib import import_module
import inspect
import shutil
import json

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torchvision import datasets, transforms
from torchcurv.optim import SecondOrderOptimizer, VIOptimizer
from torchcurv.utils import Logger

DATASET_CIFAR10 = 'CIFAR-10'
DATASET_CIFAR100 = 'CIFAR-100'


def train(model, device, train_loader, optimizer, epoch, args, logger):
    model.train()

    total_correct = 0
    loss = None
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch
    lr = optimizer.param_groups[0]['lr']

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        for i, param_group in enumerate(optimizer.param_groups):
            p = parameters_to_vector(param_group['params'])
            attr = 'p_pre_{}'.format(i)
            setattr(optimizer, attr, p.clone())

        # update params
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            return loss, output

        loss, output = optimizer.step(closure=closure)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        loss = loss.item()
        total_correct += correct

        iteration = base_num_iter + batch_idx + 1
        total_data_size += len(data)

        if batch_idx % args.log_interval == 0:
            accuracy = 100. * total_correct / total_data_size
            elapsed_time = logger.elapsed_time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, '
                  'Accuracy: {:.0f}/{} ({:.2f}%), '
                  'Elapsed Time: {:.1f}s'.format(
                  epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch,
                  loss, total_correct, total_data_size, accuracy, elapsed_time))

            # write to log
            log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                   'accuracy': accuracy, 'loss': loss, 'lr': lr}

            for i, param_group in enumerate(optimizer.param_groups):
                p = parameters_to_vector(param_group['params'])
                attr = 'p_pre_{}'.format(i)
                p_pre = getattr(optimizer, attr)
                p_norm = p.norm().item()
                upd_norm = p.sub(p_pre).norm().item()

                group_log = {'p_norm': p_norm, 'upd_norm': upd_norm}
                log[i] = group_log

            logger.write(log)

    accuracy = 100. * total_correct / epoch_size

    return accuracy, loss


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

    return test_accuracy, test_loss


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_CIFAR10, DATASET_CIFAR100], default=DATASET_CIFAR10,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
    parser.add_argument('--random_crop', action='store_true',
                        help='[data augmentation] random crop')
    parser.add_argument('--random_horizontal_flip', action='store_true',
                        help='[data augmentation] random horizontal flip')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default='models/lenet.py',
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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--scheduler_name', type=str, default=None,
                        help='name of the learning rate scheduler')
    parser.add_argument('--scheduler_args', type=json.loads, default=None,
                        help='[JSON] arguments for the scheduler')
    # Options
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
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
    parser.add_argument('--config', default=None,
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)

    def extract_kwargs(func, target):
        if target is None:
            return {}

        keys = list(inspect.signature(func).parameters.keys())
        kwargs = {}
        for key, val in target.items():
            if key in keys:
                kwargs[key] = val
        return kwargs

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup data augmentation & data pre processing
    train_transforms, test_transforms = [], []
    if args.random_crop:
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if args.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    if args.normalizing_data:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transforms.append(normalize)
        test_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    # Setup data loader for CIFAR-10/CIFAR-100
    if args.dataset == DATASET_CIFAR10:
        num_classes = 10
        dataset_class = datasets.CIFAR10
    else:
        num_classes = 100
        dataset_class = datasets.CIFAR100

    train_dataset = dataset_class(
        root=args.root, train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(
        root=args.root, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # Setup model
    _, ext = os.path.splitext(args.arch_file)
    dirname = os.path.dirname(args.arch_file)

    if dirname == '':
        module_path = args.arch_file.replace(ext, '')
    elif dirname == '.':
        module_path = os.path.basename(args.arch_file).replace(ext, '')
    else:
        module_path = '.'.join(os.path.split(args.arch_file)).replace(ext, '')

    module = import_module(module_path)
    arch_class = getattr(module, args.arch_name)

    arch_kwargs = extract_kwargs(arch_class.__init__, args.arch_args)
    arch_kwargs['num_classes'] = num_classes

    model = arch_class(**arch_kwargs)
    model = model.to(device)

    # Setup optimizer
    if args.optim_name == SecondOrderOptimizer.__name__:
        optim_class = SecondOrderOptimizer
    elif args.optim_name == VIOptimizer.__name__:
        optim_class = VIOptimizer
    else:
        optim_class = getattr(torch.optim, args.optim_name)

    optim_kwargs = extract_kwargs(optim_class.__init__, args.optim_args)
    optim_kwargs['lr'] = args.lr

    if optim_class in [SecondOrderOptimizer, VIOptimizer]:
        optimizer = optim_class(model, **optim_kwargs, **args.curv_args)
    else:
        optimizer = optim_class(model.parameters(), **optim_kwargs)

    # Setup lr scheduler
    scheduler_kwargs = {}
    if args.scheduler_name is None:
        scheduler = None
    else:
        scheduler_class = getattr(torch.optim.lr_scheduler, args.scheduler_name)
        scheduler_kwargs = extract_kwargs(scheduler_class.__init__, args.scheduler_args)
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    # Load checkpoint
    if args.resume is not None:
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.resume), 'Error: no checkpoint file found'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1

    # All config
    print('===========================')
    for key, val in vars(args).items():
        if key == 'dataset':
            print('{}: {}'.format(key, val))
            print('train data size: {}'.format(len(train_loader.dataset)))
            print('test data size: {}'.format(len(test_loader.dataset)))
        elif key == 'arch_args':
            print('   {}'.format(arch_kwargs))
        elif key == 'optim_args':
            print('   {}'.format(optim_kwargs))
        elif key == 'scheduler_args':
            print('   {}'.format(scheduler_kwargs))
        else:
            print('{}: {}'.format(key, val))
    print('===========================')

    # Copy this file to args.out
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    shutil.copy(os.path.realpath(__file__), args.out)

    # Setup logger
    logger = Logger(args.out, args.log_file_name)
    logger.start()

    # Run training
    for epoch in range(start_epoch, args.epochs + 1):

        # update learning rate
        if scheduler is not None:
            scheduler.step(epoch - 1)

        # train
        accuracy, loss = train(model, device, train_loader, optimizer, epoch, args, logger)

        # test
        test_accuracy, test_loss = test(model, test_loader, device)

        # write to log
        iteration = epoch * len(train_loader)
        log = {'epoch': epoch, 'iteration': iteration,
               'accuracy': accuracy, 'loss': loss,
               'test_accuracy': test_accuracy, 'test_loss': test_loss,
               'lr': optimizer.param_groups[0]['lr']}
        logger.write(log)

        # save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            path = os.path.join(args.out, '{}_{}_epoch{}.pt'.format(
                args.dataset, args.arch_name, epoch))
            data = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            torch.save(data, path)


if __name__ == '__main__':
    main()

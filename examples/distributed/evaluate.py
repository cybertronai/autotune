import os
import argparse
from importlib import import_module
import json
import pickle

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchcurv.optim import *

DATASET_CIFAR10 = 'CIFAR-10'
DATASET_CIFAR100 = 'CIFAR-100'
DATASET_IMAGENET = 'ImageNet'
DATASET_IMAGENET10 = 'ImageNet10'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint path for evaluation')
    parser.add_argument('--out', type=str,
                        help='dir to save output files')
    parser.add_argument('--val_num_mc_samples', type=int, default=None,
                        help='number of MC samples for validation')
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_CIFAR10, DATASET_CIFAR100, DATASET_IMAGENET, DATASET_IMAGENET10],
                        default=DATASET_CIFAR10,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--train_root', type=str, default=None,
                        help='root of train dataset')
    parser.add_argument('--val_root', type=str, default=None,
                        help='root of validate dataset')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
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
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--config', default=None,
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup data pre processing
    val_transforms = []

    if args.dataset in [DATASET_CIFAR10, DATASET_CIFAR100]:
        # CIFAR-10/100
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        # ImageNet
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        val_transforms.append(transforms.Resize(256))
        val_transforms.append(transforms.CenterCrop(224))

    val_transforms.append(transforms.ToTensor())

    if args.normalizing_data:
        val_transforms.append(normalize)

    val_transform = transforms.Compose(val_transforms)

    # Setup data loader
    if args.dataset in [DATASET_IMAGENET, DATASET_IMAGENET10]:
        # ImageNet
        if args.dataset == DATASET_IMAGENET:
            num_classes = 1000
        else:
            num_classes = 10

        train_root = args.root if args.train_root is None else args.train_root
        val_root = args.root if args.val_root is None else args.val_root
        train_dataset = datasets.ImageFolder(root=train_root)
        val_dataset = datasets.ImageFolder(root=val_root, transform=val_transform)
    else:
        if args.dataset == DATASET_CIFAR10:
            # CIFAR-10
            num_classes = 10
            dataset_class = datasets.CIFAR10
        else:
            # CIFAR-100
            num_classes = 100
            dataset_class = datasets.CIFAR100

        train_dataset = dataset_class(
            root=args.root, train=True, download=args.download)
        val_dataset = dataset_class(
            root=args.root, train=False, download=args.download, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    # Setup model
    if args.arch_file is None:
        arch_class = getattr(models, args.arch_name)
    else:
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

    arch_kwargs = {} if args.arch_args is None else args.arch_args
    arch_kwargs['num_classes'] = num_classes

    model = arch_class(**arch_kwargs)
    model = model.to(device)

    # Setup optimizer
    optim_kwargs = {} if args.optim_args is None else args.optim_args

    # Setup optimizer
    if args.optim_name in [SecondOrderOptimizer.__name__, DistributedSecondOrderOptimizer.__name__]:
        optimizer = SecondOrderOptimizer(model, **optim_kwargs, **args.curv_args)
    elif args.optim_name in [VIOptimizer.__name__, DistributedVIOptimizer.__name__]:
        optimizer = VIOptimizer(model, dataset_size=len(train_dataset), seed=args.seed,
                                **optim_kwargs, **args.curv_args)
        if args.val_num_mc_samples is not None:
            optimizer.defaults['val_num_mc_samples'] = args.val_num_mc_samples
    else:
        optim_class = getattr(torch.optim, args.optim_name)
        optimizer = optim_class(model.parameters(), **optim_kwargs)

    # Load checkpoint
    assert os.path.exists(args.checkpoint), 'Error: no checkpoint file found'
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # All config
    print('===========================')
    print('[Evaluation]')
    for key, val in vars(args).items():
        if key == 'dataset':
            print('{}: {}'.format(key, val))
            print('train data size: {}'.format(len(train_dataset)))
            print('val data size: {}'.format(len(val_loader.dataset)))
        else:
            print('{}: {}'.format(key, val))
    print('===========================')

    model.eval()
    val_loss = 0
    correct = 0

    prediction = {'correct': [], 'confidence': []}

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):

            data, target = data.to(device), target.to(device)

            print('\revaluate [{}/{}]'.format(i+1, len(val_loader)), end='')

            if isinstance(optimizer, VIOptimizer):
                optimizer.set_random_seed()
                output = optimizer.prediction(data)
            else:
                output = model(data)

            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            prediction['correct'].extend(target.eq(output.argmax(dim=1)).tolist())
            prob = F.softmax(output, dim=1)
            top1_prob, _ = torch.max(prob, dim=1)
            prediction['confidence'].extend(top1_prob.tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    print('\nEval: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), val_accuracy))

    filepath = os.path.join(args.out, 'prediction.pickle')
    with open(filepath, 'wb') as f:
        pickle.dump(prediction, f)


if __name__ == '__main__':
    main()

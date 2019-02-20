'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import optimizers

import json
import os
import argparse
import time

from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

#data
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset CIFAR10 or CIFAR100  (default: CIFAR10)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=100,
                    help='batch size for testing (default: 100)')
parser.add_argument('--epoch', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--model', type=str, default='lenet',
                    help='network model  (default: lenet)')
parser.add_argument('--optim', type=str, default='kfac',
                    help='optimizer  (default: kfac)')
parser.add_argument('--random_crop', action='store_true',
                    help='[data augmentation] random crop')
parser.add_argument('--random_horizontal_flip', action='store_true',
                    help='[data augmentation] random horizontal flip')
parser.add_argument('--normalizing_data', action='store_true',
                    help='[data pre processing] normalizing data')
parser.add_argument('--train_subset', type=int, default=None,
                    help='train sub set (default: None)')
#optimizer
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay (default: 0)')
parser.add_argument('--l2_reg', type=float, default=0,
                    help='l2 regularization (default: 0)')
parser.add_argument('--cov_ema_decay', type=float, default=0.99,
                    help='cov_ema_decay (default: 0.99)')
parser.add_argument('--damping', type=float, default=0.01,
                    help='damping (default: 0.01)')
parser.add_argument('--pi_type', type=str, default='trace_norm',
                    help='pi(for damping) norm type  (default: trace_norm)')
parser.add_argument('--std_scale', type=float, default=1e-4,
                    help='std_scale(for kfacvi) (default: 1e-4)')
parser.add_argument('--num_samples', type=int, default=1,
                    help='sample num (default: 1)')
parser.add_argument('--test_num_samples', type=int, default=None,
                    help='test sample num (default: None)')

#lr scheduler
parser.add_argument('--scheduler_type', type=str, default='standard',
                    help='scheduler type  (default: standard)')
parser.add_argument('--lr_decay', type=float, default=0,
                    help='lr decay (default: 0)')
parser.add_argument('--lr_decay_epoch', type=int, default=20,
                    help='lr decay epoch (default: 20)')
parser.add_argument('--polynomial_decay_rate', type=float, default=0,
                    help='polynomial decay rate (default: 0)')
parser.add_argument('--warmup_initial_lr', type=float, default=None,
                    help='warm up initial lr (default: None)')
parser.add_argument('--warmup_max_count', type=int, default=None,
                    help='warm up iteration (default: None)')

#options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--out', '-o', default=None,
                    help='Directory to output the result')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--progress_bar_off', action='store_true',
                    help='without progress_bar (for slurm job)')
parser.add_argument('--config', default=None,
                    help='config file path')

args = parser.parse_args()

#load config file
if args.config is not None:
    config = json.load(open(args.config))
    dict_args = vars(args)
    for key,item in config.items():
        dict_args[key] = item

#show config
print(vars(args))

#set seed
if args.seed is not None:
    torch.manual_seed(args.seed)


#for executing with job file
if args.progress_bar_off == False:
    from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

# data augmentation & data pre processing
print('==> Preparing data..')
DA_train = []
DA_test = []
if args.random_crop == True:
    DA_train.append(transforms.RandomCrop(32, padding=4))
if args.random_horizontal_flip == True:
    DA_train.append(transforms.RandomHorizontalFlip())
DA_train.append(transforms.ToTensor())
DA_test.append(transforms.ToTensor())
if args.normalizing_data == True:
    DA_train.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    DA_test.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

transform_train = transforms.Compose(DA_train)

transform_test = transforms.Compose(DA_test)

#data set
if args.dataset == 'CIFAR10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if args.dataset == 'CIFAR100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
if args.train_subset is not None:
    trainset = torch.utils.data.Subset(trainset,range(args.train_subset))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#model
print('==> Building model..')
if args.dataset == 'CIFAR10':
    net = {
            'vgg':VGG,
            'resnet':ResNet18,
            'preact_resnet':PreActResNet18,
            'googlenet':GoogLeNet,
            'densnet':DenseNet121,
            'resnext':ResNeXt29_2x64d,
            'mobilenet':MobileNet,
            'moblienetv2':MobileNetV2,
            'dpn':DPN92,
            'shufflenet':ShuffleNetG2,
            'senet':SENet18,
            'lenet':LeNet,
            'alexnet':AlexNet,
    }[args.model]()
if args.dataset == 'CIFAR100':
    net = ResNet18(num_classes)
net = net.to(device)

#data parallel
'''
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    print('Data parallel')
#'''

# Load checkpoint.
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(),lr=args.lr)
elif args.optim == 'kfac':
    optimizer = optimizers.KFAC(net,lr=args.lr, momentum=args.momentum,damping=args.damping,cov_ema_decay=args.cov_ema_decay,weight_decay=args.weight_decay,l2_reg=args.l2_reg,pi_type=args.pi_type)
elif args.optim == 'kfacvi':
    optimizer = optimizers.KFAC_VI(net,lr=args.lr,momentum=args.momentum,damping=args.damping,cov_ema_decay=args.cov_ema_decay,weight_decay=args.weight_decay,l2_reg=args.l2_reg,pi_type=args.pi_type,std_scale=args.std_scale,num_samples = args.num_samples)


polynomial_start_iter = 0
if (args.warmup_initial_lr is not None) and (args.warmup_max_count is not None):
    warmup_scheduler = optimizers.lr_scheduler_iter.GradualWarmupLR(optimizer=optimizer,initial_lr=args.warmup_initial_lr,max_count=args.warmup_max_count)
    polynomial_start_iter = args.warmup_max_count
if args.scheduler_type ==  'standard':
    scheduler = optimizers.lr_scheduler.StepLR(optimizer=optimizer,step_size=args.lr_decay_epoch,gamma=args.lr_decay)
if args.scheduler_type ==  'polynomial':
    max_count = (args.epoch+1)*len(trainloader)
    #max_count = args.epoch*len(trainloader)
    scheduler = optimizers.lr_scheduler_iter.PolynomialDecayLR(optimizer=optimizer,rate=args.polynomial_decay_rate,start_iter=polynomial_start_iter,max_count=max_count)


if args.out != None:
    #output params json file
    json_params = vars(args)
    result_path = './'+args.out
    os.makedirs(result_path,exist_ok=True)
    f_params = open(result_path+'/params.json','w')
    json.dump(json_params,f_params,indent=4)
    f_params.close()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.scheduler_type ==  'polynomial':
            scheduler.step()
        if (args.warmup_initial_lr is not None) and (args.warmup_max_count is not None):
            warmup_scheduler.step()
        inputs, targets = inputs.to(device), targets.to(device)
        if args.optim != 'kfacvi':
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            outputs,loss = optimizer.step(inputs,targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        train_loss_mean = train_loss/(batch_idx+1)
        acc = 100.*correct/total
        
        if args.progress_bar_off == False:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss_mean, acc, correct, total))

    return train_loss_mean,acc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.optim != 'kfacvi':
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            else:
                outputs,loss = optimizer.inference(inputs,targets,args.test_num_samples)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            test_loss_mean = test_loss/(batch_idx+1)
            acc = 100.*correct/total

            if args.progress_bar_off == False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss_mean, acc, correct, total))

    return test_loss_mean,acc
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

if args.out != None:
    print(result_path)
    iter_per_epoch = len(trainloader)
    total_iter = 0
    start_time = time.time()
    log = []
for epoch in range(start_epoch, start_epoch+args.epoch):
    if args.scheduler_type ==  'standard':
        scheduler.step()
    train_loss,train_acc = train(epoch)
    test_loss,test_acc = test(epoch)
    if args.out != None:
        total_iter += iter_per_epoch
        elapsed_time = time.time() - start_time
        l = {'train_loss':train_loss,'train_accuracy':train_acc,'test_loss':test_loss,'test_accuracy':test_acc,'epoch':epoch,'iteration':total_iter,'elapsed_time':elapsed_time}
        log.append(l)
        f_log = open(result_path+'/log','w')
        json.dump(log,f_log,indent=4)
        f_log.close()

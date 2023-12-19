from __future__ import print_function
import sys
import argparse
import os
import random
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from utils import mkdir_p, savefig
from utils.tools import train, test, save_checkpoint, adjust_learning_rate
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from utils.focal_loss import FocalLoss

def parser_args():
    parser = argparse.ArgumentParser(description='MetDIT Training & Testing via PyTorch')
    # Datasets
    parser.add_argument('-d', '--dataset', default='CA_01',
                        type=str,
                        help='used dataset for model training.')
    parser.add_argument('-j', '--workers', default=1,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 1).')
    parser.add_argument('-r', '--root_path',
                        default='./dataset', type=str,
                        help='the root path of training or testing images.')
    # Optimization options
    parser.add_argument('-ep', '--epochs', default=60,
                        type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0,
                        type=int, metavar='N',
                        help='manual epoch number (useful on restarts).')
    parser.add_argument('--train-batch', default=32,
                        type=int, metavar='N',
                        help='the batch-size for model training')
    parser.add_argument('--test-batch', default=16,
                        type=int, metavar='N',
                        help='the batch-size for model testing.')
    parser.add_argument('--lr', '--learning-rate', default=0.0001,
                        type=float, metavar='LR',
                        help='initial learning rate.')
    parser.add_argument('--optimizer_type', default="Adam",
                        type=str,
                        help='optimizer_type')
    parser.add_argument('--drop', '--dropout', default=0.3,
                        type=float, metavar='Dropout',
                        help='Dropout ratio for model training (Dropout 1D).')
    parser.add_argument('--schedule', type=int,
                        nargs='+', default=[50, 80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float,
                        default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M',
                        help='momentum of optimizer.')
    parser.add_argument('--weight-decay', '--wd', default=5e-4,
                        type=float, metavar='W',
                        help='weight decay (default: 1e-4).')
    parser.add_argument('--loss', type=str, default='ce', 
                        choices=['ce', 'focal'], 
                        help='the loss function for model training.')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint',
                        type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture
    parser.add_argument('-a', '--arch', default='ResNet18',
                        type=str, help='deep learning model for training')
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=1,
                        help='manual seed for model training.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set.')
    # # Device options
    # parser.add_argument('--use_cuda', default=True, type=bool,
    #                     help='Use CUDA to train model')

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    state = {k: v for k, v in args._get_kwargs()}

    # logging configuration
    # title = 'bio-ai-' + args.arch
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_rec = open('log.txt', 'w')
    logging.info("==> Training start ... ")

    # Evaluation Metric
    best_acc = 0  # best test accuracy
    # GPU devices
    if not torch.cuda.is_available():
        raise ValueError('Please use GPU for model training.')
    
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    
    # Random seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    logging.info('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set_path = os.path.join(args.root_path, args.dataset, 'train')
    test_set_path = os.path.join(args.root_path, args.dataset, 'test')
    num_classes = 2

    # train_data_loader
    # train_set_path = './dataset/CA_Dataset_pre/train'
    # train_set_path = './dataset/CA_1220/P2/train'
    trainset = ImageFolder(root=train_set_path, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    # test_data_loader
    # test_set_path = './dataset/CA_Dataset_pre/test'
    # test_set_path = './dataset/CA_1220/P2/test'
    testset = ImageFolder(root=test_set_path, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    logging.info("==> Creating model '{}'".format(args.arch))
    if args.arch == 'ResNet18':
        model = ResNet18(num_classes=num_classes)
        checkpoint = torch.load('./pretrain/pretrain-r18.pth')
        model.load_state_dict(checkpoint, strict=False)
    elif args.arch == 'ResNet34':
        model = ResNet34(num_classes=num_classes)
        checkpoint = torch.load('./pretrain/pretrain-r34.pth')
        model.load_state_dict(checkpoint, strict=False)
    elif args.arch == 'ResNet50':
        model = ResNet50(num_classes=num_classes)
        checkpoint = torch.load('./pretrain/pretrain-r50.pth')
        model.load_state_dict(checkpoint, strict=False)
    elif args.arch == 'ResNet101':
        model = ResNet101(num_classes=num_classes)
        checkpoint = torch.load('./pretrain/pretrain-r101.pth')
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise NotImplementedError('Not support {}.')
    
    model = model.to(device)

    logging.info('==> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # criterion = nn.CrossEntropyLoss()
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
        logging.info('==> CrossEntropy Loss for model training.')
    elif args.loss == 'focal':
        criterion = FocalLoss()
        logging.info('==> Focal Loss for model training.')
    else:
        raise NotImplementedError('Not support the criterion of {}'.format(args.loss))
        
    if args.optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        logging.info('==> Using SGD as optimizer.')
    elif args.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        logging.info('==> Using Adam as optimizer.')
    else:
        raise NotImplementedError('Not support the optimizer of {}'.format(args.optimizer_type))
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, 0.1, -1)

    # Resume
    if args.resume:
        # Load checkpoint.
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Train and Evaluation
    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        logging.info('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_scheduler.get_last_lr()[0]))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, device)
        test_loss, test_acc = test(testloader, model, criterion, device)
        lr_scheduler.step()
        
        save_line = ('Epoch: {}, LR: {}, Train_Loss: {:.6f}, Test_Loss: {:.4f}, Train_Acc: {:.4f}%, Test_Acc: {:.4f}%'.
                     format(epoch + 1, state['lr'], train_loss, test_loss, train_acc, test_acc))
        logging.info(save_line)
        log_rec.write(save_line)
        log_rec.write('\n')

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint, filename='{}.pth.tar'.format(epoch + 1))

    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    logging.info('Best acc: {}%'.format(best_acc))
    log_rec.write('Best acc: {}%'.format(best_acc))
    log_rec.close()
    

if __name__ == '__main__':
    # model training and evaluation
    main()

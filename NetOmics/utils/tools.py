from __future__ import print_function

import os
import shutil
import time
import random
import logging

import torch
import torch.nn.parallel
from utils import Logger, AverageMeter, accuracy
from utils.progress.bar import Bar

def train(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | '
                      'ETA: {eta:} | Train Loss: {loss:.4f} | Train Acc @ top1: {top1: .4f}').format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, device):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | '
                      'ETA: {eta:} | Test Loss: {loss:.4f} | Test Acc @ top1: {top1: .4f}').format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, args):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

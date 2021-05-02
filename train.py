from __future__ import print_function

import argparse
import os
import time
import random

import numpy as np
import cv2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms

import models.wideresnet as models
import dataset.cifar10 as dataset
import dataset.custom_dataset as custom_ds
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, confusion_matrix, plot_confusion_matrix, precision_recall

from session import SessionManager

def str_2_bool(s):
    if not isinstance(s, str):
        raise TypeError('Trying to convert arbitrary object to boolean.')
    
    s = s.lower()

    tlist = ['true',  't', '1', 'yes', 'y']
    flist = ['false', 'f', '0', 'no',  'n']
    if s in tlist:
        return True
    
    if s in flist:
        return False

    raise argparse.ArgumentTypeError('Expected boolean value.')

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')

# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')

# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=250,
                        help='Number of labeled data')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--enable-mixmatch', default=True, type=str_2_bool)

# Dataset options
parser.add_argument('--dataset-name', default='animals10', type=str, metavar='NAME',
                    help='name of the dataset')
parser.add_argument('--session-id', default=-1, type=int, metavar='ID',
                    help='the id of the session to be resumed')

args = parser.parse_args()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

def get_wideresnet_models(n_classes):
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=n_classes)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    return model, ema_model

def main():
    global constants
    # enable cudnn auto-tuner to find the best algorithm for the given harware
    cudnn.benchmark = True

    sm = SessionManager(dataset_name=args.dataset_name, resume_id=args.session_id)
    
    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, class_names, constants, preprocessor = \
            sm.load_dataset(args)

    model, ema_model = get_wideresnet_models(len(class_names))

    ts, writer = sm.load_checkpoint(model, ema_model, class_names, constants['lr'], constants['ema_decay'], constants['lambda_u'], constants['epochs'])

    step = 0
    # Train and val
    for epoch in range(ts.start_epoch, constants['epochs']):
        print('\nEpoch: [%d | %d]' % (epoch + 1, constants['epochs']))
        step = constants['train_iteration'] * (epoch + 1)

        if constants['enable_mixmatch']:
            train_loss = train(labeled_trainloader, unlabeled_trainloader, epoch, ts)
        else:
            train_loss = train_supervised(labeled_trainloader, epoch, ts, preprocessor)

        losses, accs, confs, names = validate_all(labeled_trainloader, val_loader, test_loader, train_loss, ts)

        tensorboard_write(writer, losses, accs, confs, names, class_names, step)

        # save model and other training variables
        sm.save_checkpoint(accs[names['validation']], epoch)

    sm.close()

def iterate_with_restart(loader, iterator):
    try:
        inputs, targets = iterator.next()
    except:
        iterator = iter(loader)
        inputs, targets = iterator.next()

    return iterator, inputs, targets


def guess_labels(inputs_u1, inputs_u2, model):
    with torch.no_grad():
        # compute guessed labels of unlabel samples
        outputs_u1 = model(inputs_u1)
        outputs_u2 = model(inputs_u2)
        p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        pt = p**(1/constants['T'])
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()

    return targets_u

def mixup(inputs_x, inputs_u1, inputs_u2, targets_x, targets_u):
    all_inputs = torch.cat([inputs_x, inputs_u1, inputs_u2], dim=0)
    all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

    l = np.random.beta(constants['alpha'], constants['alpha'])

    l = max(l, 1-l)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target

def predict_train(model, mixed_input):
    # interleave labeled and unlabed samples between batches to 
    # get correct batchnorm calculation 
    batch_size = constants['batch_size']

    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = interleave(mixed_input, batch_size)

    logits = [model(mixed_input[0])]
    for _input in mixed_input[1:]:
        logits.append(model(_input))

    # put interleaved samples back
    logits = interleave(logits, batch_size)
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    return logits_x, logits_u

def train(labeled_trainloader, unlabeled_trainloader, epoch, train_state):
    model = train_state.model
    optimizer = train_state.optimizer
    ema_optimizer = train_state.ema_optimizer
    criterion = train_state.train_criterion

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    n_classes = len(train_state.class_names)

    bar = Bar('Training', max=constants['train_iteration'])
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(constants['train_iteration']):
        labeled_train_iter, inputs_x, targets_x = \
            iterate_with_restart(labeled_trainloader, labeled_train_iter)
        unlabeled_train_iter, (inputs_u1, inputs_u2), _ = \
            iterate_with_restart(unlabeled_trainloader, unlabeled_train_iter)

        if constants['use_cuda']:
            inputs_x  = inputs_x.cuda(non_blocking = True)
            inputs_u1 = inputs_u1.cuda(non_blocking = True)
            inputs_u2 = inputs_u2.cuda(non_blocking = True)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, n_classes).scatter_(1, targets_x.view(-1,1).long(), 1)

        if constants['use_cuda']:
            targets_x = targets_x.cuda(non_blocking = True)

        targets_u = guess_labels(inputs_u1, inputs_u2, model)

        mixed_input, mixed_target = mixup(inputs_x, inputs_u1, inputs_u2,
                                        targets_x, targets_u)
        
        logits_x, logits_u = predict_train(model, mixed_input)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], 
                            logits_u, mixed_target[batch_size:], 
                            epoch+batch_idx/constants['train_iteration'])
        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=constants['train_iteration'],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    return losses.avg

def train_supervised(labeled_trainloader, epoch, train_state, preprocessor):
    model = train_state.model
    optimizer = train_state.optimizer
    ema_optimizer = train_state.ema_optimizer
    criterion = train_state.train_criterion

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    n_classes = len(train_state.class_names)

    bar = Bar('Training', max=constants['train_iteration'])
    labeled_train_iter = iter(labeled_trainloader)

    model.train()
    for batch_idx in range(constants['train_iteration']):
        labeled_train_iter, inputs_x, targets_x = \
            iterate_with_restart(labeled_trainloader, labeled_train_iter)

        if constants['use_cuda']:
            inputs_x  = inputs_x.cuda(non_blocking = True)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, n_classes).scatter_(1, targets_x.view(-1,1).long(), 1)

        if constants['use_cuda']:
            targets_x = targets_x.cuda(non_blocking = True)

        mixed_input, mixed_target = mixup(inputs_x, inputs_x, inputs_x,
                                        targets_x, targets_x)
        
        inputs  = mixed_input[:batch_size]
        targets = mixed_target[:batch_size] 
        logits  = model(inputs)

        Lx, _, _ = criterion(logits, targets, logits, targets, 
                            epoch+batch_idx/constants['train_iteration'])
        loss = Lx

        # record loss
        losses.update(loss.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=constants['train_iteration'],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()

    return losses.avg

def tensorboard_write(writer, losses, accs, confs, set_names, class_names, step):
    for loss, acc, name in zip(losses, accs, set_names):
        writer.add_scalar('losses/' + name + '_loss', loss, step)
        writer.add_scalar('accuracy/' + name + '_acc', acc, step)

    img_conf_val = plot_confusion_matrix(confs[set_names['validation']], class_names)
    writer.add_image('confusion/val', img_conf_val, step)

    val_pre, val_rec = precision_recall(confs[set_names['validation']])

    for i in range(len(val_pre)):
        writer.add_scalar('precision/val/' + class_names[i], val_pre[i], step)
        writer.add_scalar('recall/val/' + class_names[i], val_rec[i], step)

def validate_all(train_loader, val_loader, test_loader, train_loss, train_state):
    _, train_acc, train_confusion = validate(train_loader, train_state)
    val_loss, val_acc, val_confusion = validate(val_loader, train_state)
    test_loss, test_acc, test_confusion = validate(test_loader, train_state)

    losses = [train_loss, val_loss, test_loss]
    accs   = [train_acc, val_acc, test_acc]
    confs  = [train_confusion, val_confusion, test_confusion]
    names  = {'training' : 0, 'validation' : 1, 'testing' : 2}

    return losses, accs, confs, names

def validate(valloader, train_state):

    model = train_state.ema_model
    criterion = train_state.criterion

    losses = AverageMeter()
    accuracy_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    n_classes = len(train_state.class_names)
    confusion = torch.zeros(n_classes, n_classes)

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if constants['use_cuda']:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc = accuracy(outputs, targets)
            batch_confusion = confusion_matrix(outputs, targets)
            confusion += batch_confusion
            losses.update(loss.item(), inputs.size(0))
            accuracy_meter.update(acc.item(), inputs.size(0))
            
    return (losses.avg, accuracy_meter.avg, confusion)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    #print(xy[0].shape)
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()

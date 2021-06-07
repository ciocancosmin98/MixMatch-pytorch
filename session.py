import torch.optim as optim
from torch import load, save
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import os
import shutil

import dataset.custom_dataset as cds
import dataset.cifar10 as cifar10
import models.wideresnet as models

import dataset.transforms as transforms

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class MyCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, _input, target):
        target = target.long()
        return F.cross_entropy(_input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        
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

class SemiLoss(object):
    def __init__(self, lambda_u, n_epochs):
        self.lambda_u = lambda_u
        self.n_epochs = n_epochs

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, self.n_epochs)
        
class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

class TrainState:
    def __init__(self, model, ema_model, class_names, constants, session_path):
        self.best_acc = 0
        self.start_epoch = 0
        self.model = model
        self.ema_model = ema_model

        lr = constants['lr']
        alpha = constants['alpha']
        lambda_u = constants['lambda_u']
        n_epochs = constants['epochs']
        self.constants = constants
        
        self.train_criterion = SemiLoss(lambda_u, n_epochs)
        self.criterion = MyCrossEntropy()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.ema_optimizer = WeightEMA(model, ema_model, lr=lr, alpha=alpha)
        self.class_names = class_names

        training_dir = os.path.join(session_path, 'training')
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)

        self.save_path = os.path.join(training_dir, 'checkpoint.pth.tar')
        self.best_path = os.path.join(training_dir, 'best_model.pth.tar')
        self.pretrained_path = os.path.join('pretrained', 'cifar10_model.pth.tar')

    def _transfer_weights(self):
        checkpoint = load(self.pretrained_path)
        lr = self.constants['lr']
        alpha = self.constants['alpha']

        print('==> Using pretrained model from cifar10 with accuracy %.3f' % checkpoint['best_acc'])

        cifar10_model, cifar10_ema_model = get_wideresnet_models(10)
        cifar10_model.load_state_dict(checkpoint['state_dict'])
        cifar10_ema_model.load_state_dict(checkpoint['ema_state_dict'])

        self.model.block1.load_state_dict(cifar10_model.block1.state_dict())
        self.model.block2.load_state_dict(cifar10_model.block2.state_dict())
        self.model.block3.load_state_dict(cifar10_model.block3.state_dict())

        self.ema_model.block1.load_state_dict(cifar10_ema_model.block1.state_dict())
        self.ema_model.block2.load_state_dict(cifar10_ema_model.block2.state_dict())
        self.ema_model.block3.load_state_dict(cifar10_ema_model.block3.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, lr=lr, alpha=alpha)

    def _handle_resume(self, load_best=False):
        if load_best:
            resume_path = self.best_path
        else:
            resume_path = self.save_path

        if os.path.exists(resume_path):
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = load(resume_path)
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        elif self.constants['use_pretrained'] and os.path.exists(self.pretrained_path):
            self._transfer_weights()

    def is_best(self, val_acc):
        return val_acc > self.best_acc

    def _save_checkpoint(self, val_acc, epoch):
        is_best = val_acc > self.best_acc
        self.best_acc = max(val_acc, self.best_acc)

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'acc': val_acc,
            'best_acc': self.best_acc,
            'optimizer' : self.optimizer.state_dict(),
        }

        save(state, self.save_path)
        if is_best:
            shutil.copyfile(self.save_path, self.best_path)

class SessionManager:
    def __init__(self, args, sessions_root='sessions',
            datasets_root='data', session_path=None):

        dataset_name = args.dataset_name
        resume_id    = args.session_id

        self.dname = dataset_name
        self.droot = datasets_root
        self.id    = resume_id

        if session_path is None:
            session_dir = os.path.join(sessions_root, self.dname)

            if not os.path.exists(session_dir):
                os.makedirs(session_dir)

            if self.id == -1:
                next_id = 0
                for fname in os.listdir(session_dir):
                    try:
                        _id = int(fname)

                        if _id >= next_id:
                            next_id = _id + 1
                    except ValueError:
                        pass
                self.id = next_id

            session_path = os.path.join(session_dir, str(self.id))
            if not os.path.exists(session_path):
                os.makedirs(session_path)        

        self.spath = session_path

        writer_dir = os.path.join(self.spath, 'tensorboard')
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
        self.wdir = writer_dir

        preprocessing_dir = os.path.join(self.spath, 'preprocessing')
        if not os.path.exists(preprocessing_dir):
            os.makedirs(preprocessing_dir)
        self.pdir = preprocessing_dir

        self.constants = self.load_constants(args)

    
    def load_constants(self, args):
        const_path = os.path.join(self.spath, 'const.pth.tar')

        # save args if first run of the session
        if not os.path.exists(const_path):
            saved_constants = {}
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            saved_constants['use_cuda'] = torch.cuda.is_available()

            for arg, value in args._get_kwargs():
                saved_constants[arg] = value

            save(saved_constants, const_path)

        constants = load(const_path)

        # compatibility check
        for arg, value in args._get_kwargs():
            if not arg in constants:
                constants[arg] = value
        
        return constants

    def add_constant(self, name, value):
        const_path = os.path.join(self.spath, 'const.pth.tar')

        constants = load(const_path)
        constants[name] = value
        save(constants, const_path)

        self.constants = constants

    def add_constants(self, new_constants):
        const_path = os.path.join(self.spath, 'const.pth.tar')

        constants = load(const_path)
        for name, value in new_constants:
            constants[name] = value
        
        save(constants, const_path)
        self.constants = constants

    def get_constants(self):
        return self.constants

    def load_dataset(self, queue=None):
        batch_size = self.constants['batch_size']
        n_labeled  = self.constants['n_labeled']
        transforms_name = self.constants['transforms']

        if not queue is None:
            qReader = cds.ImageQueueReader(queue, self.constants['base_path'],
                            self.constants['class_names'], self.spath)

            while not qReader.tick():
                pass

            labeled_fn, unlabeled_fn, val_fn, test_fn = qReader.split()

            prep = cds.Preprocessor(labeled_fn, unlabeled_fn, val_fn, test_fn, save_dir=self.pdir, overwrite=False, size=32)

            labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = \
                    cds.load_custom(prep, batch_size, transforms_name)
        elif self.dname == "cifar10":
            # load cifar-10
            cifar_dir = os.path.join(self.droot, 'cifar10')
            labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, class_names = \
                    cifar10.load_cifar10_default(cifar_dir, batch_size, n_labeled, transforms_name)
        else:
            # load custom dataset
            labeled_fn, unlabeled_fn, val_fn, test_fn = cds.get_filenames_train_validate_test(self.dname, 
                    self.pdir, self.constants['n_labeled'], self.constants['balance_unlabeled'], self.constants['n_test_per_class'])

            prep = cds.Preprocessor(labeled_fn, unlabeled_fn, val_fn, test_fn, save_dir=self.pdir, overwrite=False, size=32)

            labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = \
                    cds.load_custom(prep, batch_size, transforms_name)

            class_names = prep.get_class_names()

        if not 'class_names' in self.constants:
            self.add_constant('class_names', class_names)

        if not 'train_iteration' in self.constants:
            min_iterations = 32

            if not self.constants['enable_mixmatch']:
                train_iteration = max(len(labeled_trainloader), min_iterations)
            else:
                train_iteration = max(max(len(labeled_trainloader), len(unlabeled_trainloader)), min_iterations)
            
            self.add_constant('train_iteration', train_iteration)

        return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader

    def load_checkpoint(self):
        class_names = self.constants['class_names']
        model, ema_model = get_wideresnet_models(len(class_names))

        self.ts = TrainState(model, ema_model, class_names, self.constants, self.spath)
        self.ts._handle_resume()
        
        self.writer = SummaryWriter(self.wdir)
        return self.ts, self.writer

    def save_checkpoint(self, val_acc, epoch):
        self.ts._save_checkpoint(val_acc, epoch)

    def close(self):
        self.writer.close()
    
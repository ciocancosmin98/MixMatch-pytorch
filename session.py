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
    def __init__(self, model, ema_model, class_names, lr, alpha, lambda_u, n_epochs, session_path):
        self.best_acc = 0
        self.start_epoch = 0
        self.model = model
        self.ema_model = ema_model
        
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
    def __init__(self, dataset_name='animals10', sessions_root='sessions',
            datasets_root='data', resume_id=-1):

        self.dname = dataset_name
        self.sroot = sessions_root
        self.droot = datasets_root
        self.id    = resume_id

        session_dir = os.path.join(self.sroot, self.dname)

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

        return constants


    def load_dataset(self, args):
        constants = self.load_constants(args)
        batch_size = constants['batch_size']
        n_labeled  = constants['n_labeled']

        if self.dname == "cifar10":
            # load cifar-10

            cifar_dir = os.path.join(self.droot, 'cifar10')
            labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, class_names = \
                    cifar10.load_cifar10_default(cifar_dir, batch_size, n_labeled)

            prep = None
        else:
            # load custom dataset
            labeled_fn, unlabeled_fn, val_fn, test_fn = cds.get_filenames_train_validate_test(self.dname, self.pdir, n_labeled=constants['n_labeled'])

            prep = cds.Preprocessor(labeled_fn, unlabeled_fn, val_fn, test_fn, save_dir=self.pdir, overwrite=False, size=32)

            labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = \
                    cds.load_custom(prep, batch_size)

            class_names = prep.get_class_names()

        if not 'train_iteration' in constants:
            min_iterations = 32

            if not constants['enable_mixmatch']:
                train_iteration = max(len(labeled_trainloader), min_iterations)
            else:
                train_iteration = max(max(len(labeled_trainloader), len(unlabeled_trainloader)), min_iterations)
            
            constants = self.add_constant('train_iteration', train_iteration)

        return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, class_names, constants, prep

    def load_checkpoint(self, model, ema_model, class_names, lr, alpha, lambda_u, n_epochs):
        self.ts = TrainState(model, ema_model, class_names, lr, alpha, lambda_u, n_epochs, self.spath)
        self.ts._handle_resume()
        
        self.writer = SummaryWriter(self.wdir)
        return self.ts, self.writer

    def save_checkpoint(self, val_acc, epoch):
        self.ts._save_checkpoint(val_acc, epoch)

    def close(self):
        self.writer.close()
    
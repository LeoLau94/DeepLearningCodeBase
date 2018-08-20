import torch
import torch.nn as nn
import os
import time
import math
from torch.optim import lr_scheduler

__all__ = [
    'Trainer',
    ]

class Trainer:

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.lr = kwargs['lr']
        self.criterion = kwargs['criterion']
        self.start_epoch = kwargs['start_epoch']
        self.epochs = kwargs['epochs']
        self.cuda = kwargs['cuda']
        self.log_interval = kwargs['log_interval']
        self.train_loader = kwargs['train_loader']
        self.validate_loader = kwargs['validate_loader']
        self.root = kwargs['root']
        self.plugins = {}
        for name in [
                'iteration',
                'epoch',
                'batch',
                'update',
                'dataforward']:
            self.plugins.setdefault(name, [])

        for p in kwargs['plugins']:
            p.register(self)
            self.plugins[p.plugin_type].append(p)

        self.writer = kwargs['writer']
        if kwargs['scheduler_class'] is None:
            kwargs['scheduler_class'] = lr_scheduler.CosineAnnealingLR
            kwargs['scheduler_kwargs'] = {'T_max': self.epochs}
        self.scheduler = kwargs['scheduler_class'](
            self.optimizer, **kwargs['scheduler_kwargs'])

    def train(self, e):
        self.model.train()
        for p in self.plugins['iteration']:
            p.reset()

        time_stamp = time.time()

        for batch_idx, (data, label) in enumerate(self.train_loader):

            if self.cuda:
                data, label = data.cuda(), label.cuda(async=True)

            output, loss = self.plugins['dataforward'][0].call(data, label)

            for p in self.plugins['iteration']:
                p.call(output, label, loss)

            self.optimizer.zero_grad()
            loss.backward()
            for p in self.plugins['update']:
                p.call()
            self.optimizer.step()

            if (batch_idx + 1) % self.log_interval == 0:
                for p in self.plugins['iteration']:
                    p.logger(e=e)

        for p in self.plugins['iteration']:
            p.logger(train=False)

        print('Training Time: {}'.format(time.time() - time_stamp))

        results = {}
        for p in self.plugins['iteration']:
            results[p.name] = p.getState()

        return results

    def validate(self):
        self.model.eval()
        for p in self.plugins['iteration']:
            p.reset()

        time_stamp = time.time()
        for data, label in self.validate_loader:
            if self.cuda:
                data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                output, loss = self.plugins['dataforward'][0].call(
                    data, label, train=False)

            for p in self.plugins['iteration']:
                p.call(output, label, loss)

        for p in self.plugins['iteration']:
            p.logger(train=False)
        print('Validation Time: {}'.format(time.time() - time_stamp))

        results = {}
        for p in self.plugins['iteration']:
            results[p.name] = p.getState()

        return results

    def start(self):

        if not os.path.exists(self.root):
            os.makedirs(self.root)
        print(self.model)
        if self.cuda:
            self.model.cuda()
            #self.model = nn.dataparallel(self.model)
            print("using gpu: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
        print('-----start training-----\n')
        best_precision = 0
        for e in range(self.start_epoch, self.epochs):
            self.scheduler.step()
            train_results = self.train(e)
            validation_results = self.validate()

            for p in self.plugins['epoch']:
                p.call(train_results, validation_results, e)

            top1Acc = validation_results['TopK Accuracy']['Top 1']
            is_best = top1Acc > best_precision
            best_precision = top1Acc if top1Acc > best_precision else best_precision
            training_state = {
                'start_epoch': e,
            }
            self.save_checkpoint(
                training_state,
                is_best
                )
        print("-----Training Completed-----\n")
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

    def save_checkpoint(self, training_state, is_best):
        state = {
            'cuda': self.cuda,
            'start_epoch': training_state['start_epoch'] + 1,
            'epochs': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        file = os.path.join(self.root, 'checkpoint' + '.pkl')
        file_best = os.path.join(self.root, 'best_precision_model_params.pkl')
        torch.save(state, file)
        if is_best:
            torch.save(self.model.state_dict(), file_best)

    def load_checkpoint(self, root, is_resumed=False):
        if os.path.isfile(root):
            print("Loading checkpoint at '{}'".format(root))
            checkpoint = torch.load(root)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.cuda = checkpoint['cuda']
            if is_resumed:
                self.start_epoch = checkpoint['start_epoch']
                self.epochs = checkpoint['epochs']
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
            return True
        else:
            print("'{}' doesn't exist!".format(root))
            return False

    def resume(self, root):
        if self.load_checkpoint(root, is_resumed=True):
            print('Successfully resume')
        else:
            print('Failed to resume')

    def adjust_lr_cosine(self, epoch, batch, nBatch):
        T_total = self.epochs * nBatch
        T_cur = (epoch % self.epochs) * nBatch + batch
        lr = 0.5 * self.lr * (1 + math.cos(math.pi * T_cur / T_total))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
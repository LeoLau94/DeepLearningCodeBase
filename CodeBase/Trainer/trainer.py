import torch
import os
import time
import math

__all__ = [
    'Trainer',
    ]


class Trainer:

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = kwargs['criterion']
        self.start_epoch = 0
        self.epochs = kwargs['epochs']
        self.cuda = kwargs['cuda']
        self.log_interval = kwargs['log_interval']
        self.save_interval = kwargs['save_interval']
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
            print("using gpu: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
        print('-----start training-----\n')
        best_precision = 0
        for e in range(self.start_epoch, self.epochs + 1):
            self.scheduler.step()
            train_results = self.train(e)
            validation_results = self.validate()

            for p in self.plugins['epoch']:
                p.call(train_results, validation_results, e)

            top1Acc = validation_results['TopK Accuracy']['Top 1']
            is_best = top1Acc > best_precision
            if is_best:
                best_precision = top1Acc
                training_state = {
                    'start_epoch': e,
                    'epochs': self.epochs,
                    'cuda': self.cuda,
                    'precision@1': best_precision,
                }
                self.save_checkpoint(
                    training_state,
                    is_best
                )
            if e % self.save_interval == 0:
                training_state = {
                    'start_epoch': e,
                    'epochs': self.epochs,
                    'cuda': self.cuda,
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
            'training_state_dict': training_state,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if is_best:
            checkpoint_best = os.path.join(
                self.root, 'best_precision_checkpoint.pkl')
            torch.save(state, checkpoint_best)
            torch.save(
                self.model.state_dict(),
                'best_precision_model_params.pkl')
        else:
            checkpoint = os.path.join(
                self.root, '.'.join(
                    str(training_state['start_epoch']),
                    'checkpoint', 'pkl'))
            torch.save(state, checkpoint)

    def load_checkpoint(self, root, is_resumed=False):
        if os.path.isfile(root):
            print("Loading checkpoint at '{}'".format(root))
            checkpoint = torch.load(root)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.cuda = checkpoint['training_state_dict']['cuda']
            if is_resumed:
                self.start_epoch = checkpoint[
                    'training_state_dict']['start_epoch']
                self.epochs = checkpoint['training_state_dict']['epochs']
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

import torch
import os
import time
import math

__all__ = [
    'Trainer',
    ]


class Trainer:

    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.optimizer = kwargs.get('optimizer')
        self.scheduler = kwargs.get('scheduler')
        self.criterion = kwargs.get('criterion')
        self.start_epoch = 1
        self.epochs = kwargs.get('epochs', 100)
        self.cuda = kwargs.get('cuda', True)
        self.log_interval = kwargs.get('log_interval', 100)
        self.save_interval = kwargs.get('save_interval', 10)
        self.train_loader = kwargs.get('train_loader')
        self.validate_loader = kwargs.get('validate_loader')
        self.classes = kwargs.get('classes', self.train_loader.dataset.classes)
        self.root = kwargs.get('root')
        self.plugins = {}
        for name in [
                'iteration',
                'epoch',
                'batch',
                'update',
                'dataforward']:
            self.plugins.setdefault(name, [])

        for p in kwargs.get('plugins'):
            p.register(self)
            self.plugins[p.plugin_type].append(p)
        self.writer = kwargs.get('writer')
        if self.cuda:
            self.model.cuda()
            print("using gpu: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

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
            training_state = {
                'start_epoch': e,
                'epochs': self.epochs,
                'cuda': self.cuda,
                'precisin@1': top1Acc.item()
            }
            if is_best:
                best_precision = top1Acc
                self.save_checkpoint(
                    training_state,
                    is_best
                )
            if e % self.save_interval == 0:
                self.save_checkpoint(
                    training_state,
                    False
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
            checkpoint = os.path.join(
                self.root, 'best_performance_checkpoint.pkl')
            model_params_path = os.path.join(
                self.root, 'best_performance_model_params.pkl')
            torch.save(
                self.model.state_dict(),
                model_params_path)
        else:
            checkpoint = os.path.join(
                self.root, '.'.join([
                    str(training_state['start_epoch']),
                    'checkpoint', 'pkl']))
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
                if self.cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
            return True
        else:
            print("'{}' doesn't exist!".format(root))
            return False

    def resume(self, root):
        if self.load_checkpoint(root, is_resumed=True):
            print('Successfully resume')
        else:
            raise RuntimeError("Failed to resume")

    def adjust_lr_cosine(self, epoch, batch, nBatch):
        T_total = self.epochs * nBatch
        T_cur = (epoch % self.epochs) * nBatch + batch
        lr = 0.5 * self.lr * (1 + math.cos(math.pi * T_cur / T_total))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

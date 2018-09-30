import torch
import torch.nn as nn
from datetime import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.total = 0

    def update(self, val, delta):
        self.val = val
        self.count += val * delta
        self.total += delta
        if self.total > 0:
            self.avg = self.count / self.total


class Plugin:

    def __init__(self):
        pass

    def __current_time__(self):
        current_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S')
        return current_time

    def register(self, trainer):
        self.trainer = trainer

    def logger(self):
        print('------------%s--------------' % self.name)

    def call(self):
        raise NotImplementedError


class IterationMonitor(Plugin):

    def __init__(self):
        self.plugin_type = 'iteration'


class EpochMonitor(Plugin):

    def __init__(self):
        self.plugin_type = 'epoch'


class BatchMonitor(Plugin):

    def __init__(self):
        self.plugin_type = 'batch'


class UpdateMonitor(Plugin):

    def __init__(self):
        self.plugin_type = 'update'


class DataForward(Plugin):

    def __init__(self, train_forward, validation_forward=None, **kwargs):
        self.plugin_type = 'dataforward'
        self.train_forward = train_forward
        self.validation_forward = validation_forward
        self.kwargs = kwargs

    def call(self, data, target, train=True):
        if train:
            return self.train_forward(self, data, target)
        elif self.validation_forward:
            return self.validation_forward(self, data, target)
        else:
            return self.train_forward(self, data, target)


class ModelGradHandler(UpdateMonitor):

    def __init__(self, lambda_=None, **kwargs):
        super(ModelGradHandler, self).__init__()
        self.lambda_ = lambda_
        self.kwargs = kwargs

    def register(self, trainer):
        self.model = trainer.model

    def call(self):
        self.lambda_(self)


class DataMonitor(IterationMonitor):

    def call(self, output, target, loss):
        if not isinstance(output, tuple):
            self.__handle__(output.data, target.data, loss.item())
        else:
            self.__handle__(output[0].data, target.data, loss.item())

    def __handle__(self, output, target):
        raise NotImplementedError

    def getState(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class LossMonitor(DataMonitor):
    name = 'Loss'

    def __init__(self):
        super(LossMonitor, self).__init__()
        self.batch_idx = 0
        self.loss = AverageMeter()
        self.lr = None

    def __handle__(self, output, target, loss):
        self.batch_idx += 1
        self.batch_size = output.size(0)
        self.loss.update(loss, output.size(0))

    def logger(self, e=None, train=True):
        if train:
            self.lr = iter(self.trainer.optimizer.param_groups).__next__()['lr']
            print('=======================================')
            print('{}\t Epoch: {} [{}/{}]\t'
                  'Loss: {loss.val:.6f}\t'
                  'LR: {}'
                  .format(
                      self.__current_time__(),
                      e,
                      self.batch_idx * self.batch_size,
                      len(self.trainer.train_loader.dataset),
                      self.lr,
                      loss=self.loss,
                  ))

        else:
            if self.trainer.model.training:
                self.lr = iter(
                    self.trainer.optimizer.param_groups).__next__()['lr']
                print(
                    '==================Training Epoch Summary=====================')
                print('{}\t'
                      'Loss: {loss.avg:.6f}\t'
                      'LR: {}'
                      .format(
                        self.__current_time__(),
                        self.lr,
                        loss=self.loss,
                          ))
            else:
                print(
                    '==================Validation Summary=====================')
                print('{}\t'
                      'Loss: {loss.avg:.6f}\t'
                      .format(
                          self.__current_time__(),
                          loss=self.loss,
                      ))

    def getState(self):
        State = {'Loss': self.loss.avg}
        return State

    def reset(self):
        self.loss.reset()
        self.batch_idx = 0


class TopKAccuracy(DataMonitor):
    name = 'TopK Accuracy'

    def __init__(self, topk=(1,)):
        super(TopKAccuracy, self).__init__()
        self.topk = topk
        self.topAcc = [AverageMeter() for i in self.topk]

    def __handle__(self, output, target, loss):
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            val = correct_k.mul_(100.0 / batch_size)
            self.topAcc[i].update(val, output.size(0))

    def register(self, trainer):
        super().register(trainer)
        if max(self.topk) >= len(trainer.classes):
            raise ValueError("k should be less than number of classes!")

    def logger(self, e=None, train=True):
        super().logger()
        log = ''
        if train:
            for i, k in enumerate(self.topk):
                log += 'Top {} Acc:{Acc.val:.2f}%\t'.format(
                    k, Acc=self.topAcc[i])
        else:
            for i, k in enumerate(self.topk):
                log += 'Top {} Acc:{Acc.avg:.2f}%\t'.format(
                    k, Acc=self.topAcc[i])

        print(log)

    def getState(self):
        State = {}
        for i, k in enumerate(self.topk):
            State['Top %d' % k] = self.topAcc[i].avg
        return State

    def reset(self):
        for k in self.topAcc:
            k.reset()


class ClassAccuracy(DataMonitor):
    name = 'Classes Accuracy'

    def __init__(self):
        super(ClassAccuracy, self).__init__()

    def register(self, trainer):
        super().register(trainer)
        self.classes = trainer.classes
        self.classesAcc = [AverageMeter() for c in self.classes]

    def __handle__(self, output, target, loss):
        classes_total = torch.zeros(len(self.classes)).to(torch.int)
        classes_count = torch.zeros(len(self.classes)).to(torch.int)

        target_idx = target.view(-1).to(torch.int)
        for idx in target_idx:
            classes_total[idx] += 1

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        classes_idx = pred.masked_select(correct).to(torch.int)

        for idx in classes_idx:
            classes_count[idx] += 1

        total_np = classes_total.numpy()
        those_are_zero_idx = total_np == 0
        total_np[those_are_zero_idx] = 1
        classes_acc = classes_count.mul_(100).div(classes_total)
        total_np[those_are_zero_idx] = 0

        for i, _ in enumerate(self.classes):
            self.classesAcc[i].update(
                classes_acc[i].item(), classes_total[i].item())

    def logger(self, e=None, train=True):
        super().logger()
        if train:
            for i, c in enumerate(self.classes):
                print(
                    '{} Acc:{Acc.val:.2f}%\t'.format(
                        c, Acc=self.classesAcc[i]))
        else:
            for i, c in enumerate(self.classes):
                print('{} Acc:{Acc.avg:.2f}%[{:d}/{Acc.total:d}]\t'.format(
                    c, int(self.classesAcc[i].count / 100), Acc=self.classesAcc[i]))

    def getState(self):
        State = {}
        for i, c in enumerate(self.classes):
            State['%s' % c] = self.classesAcc[i].avg
        return State

    def reset(self):
        for c in self.classesAcc:
            c.reset()


class IterationSummaryMonitor(EpochMonitor):

    def call(self, train_results, validation_results, e):
        for p in self.trainer.plugins['iteration']:
            self.trainer.writer.add_scalars(
                '%s/Training' % p.name, train_results[p.name], e)
            self.trainer.writer.add_scalars(
                '%s/Validation' % p.name,
                validation_results[
                    p.name],
                e)


class DistributionOfBNMonitor(EpochMonitor):

    def call(self, train_results, validation_results, e):
        bn_idx = 0
        for m in self.trainer.model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                bn_idx += 1
                self.trainer.writer.add_histogram(
                    'BatchNorm Layer%d Gamma' %
                    bn_idx, m.weight.clone().cpu().data.numpy(), e)
                self.trainer.writer.add_histogram(
                    'BatchNorm Layer%d Beta' %
                    bn_idx, m.bias.clone().cpu().data.numpy(), e)


class DownSampleMonitor(EpochMonitor):

    def call(self, train_results, validation_results, e):
        self.trainer.train_loader.dataset.reset()

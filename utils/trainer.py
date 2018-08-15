import torch
import torch.nn as nn
# import torch.optim as optim
from torch.autograd import Variable
# from torch.nn import functional as F
from torch.optim import lr_scheduler
import os
import time
import math
from datetime import datetime
# import numpy as np
# from scipy.integrate import quad
# from scipy import stats
# from numpy import pi, e, sqrt
# from numpy import vectorize
# from numba import float32, cfunc

import sys
sys.path.append('/home/leolau/pytorch/')
#from utils.vary_batchnorm import *
#from nets.my_vgg import *

__all__ = [
    'Trainer',
    'Network_Slimming_Trainer',
    'KD_FineTuning_Trainer',
    ]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.total = 0

    def updateWithVal(self, val, delta):
        self.val = val
        self.count += val * delta
        self.total += delta
        self.avg = self.count / self.total

    def updateWithCount(self, count=1, delta=1):
        self.count += count
        self.total += delta
        self.total = 1 if self.total == 0 else self.total
        self.avg = self.count / self.total


class VaryingBNlrClipper(object):

    def __init__(self, penalty_range):
        self.penalty_range = penalty_range

    def __call__(self, module):
        if hasattr(module, 'lr'):
            w = module.lr.data
            module.lr.data = torch.sigmoid(w) * self.penalty_range


class Trainer:

    def __init__(self, **kwargs):
        kwargs.setdefault('topk', None)
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
        self.classes = self.train_loader.dataset.classes
        if kwargs['topk'] is not None:
            self.topk = kwargs['topk']
        else:
            if len(self.classes) > 5:
                self.topk = (1, 5)
            else:
                self.topk = (1,)

        self.writer = kwargs['writer']
        self.scheduler = lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[
                self.epochs * 0.5, self.epochs * 0.75], gamma=0.1)

    def train(self, e):
        self.model.train()

        losses = AverageMeter()
        topAcc = [AverageMeter() for i in self.topk]
        classesAcc = [AverageMeter() for c in self.classes]
        time_stamp = time.time()

        for batch_idx, (data, label) in enumerate(self.train_loader):
            # lr = self.adjust_lr_cosine(e,batch_idx,len(self.train_loader))

            if self.cuda:
                data, label = data.cuda(), label.cuda(async=True)

            output = self.model(data)
            loss = self.criterion(output, label)
            prec = self.accuracy(output.data, label.data, topk=self.topk)
            classes_count, classes_total = self.classes_accuracy(
                output.data, label.data)

            losses.updateWithVal(loss.data[0], data.size(0))
            for i, _ in enumerate(self.topk):
                topAcc[i].updateWithVal(prec[i], data.size(0))
            for i, _ in enumerate(self.classes):
                classesAcc[i].updateWithCount(
                    classes_count[i].item(),
                    classes_total[i].item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (batch_idx + 1) % self.log_interval == 0:
                current_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S')
                print('=======================================')
                print('{}\t Epoch: {} [{}/{}]\t'
                      'Loss: {loss.avg:.6f}\t'
                      .format(
                        current_time,
                        e,
                        (batch_idx + 1) * len(data),
                        len(self.train_loader.dataset),
                        loss=losses,
                          ))
                print('------------Topk Accuracy--------------')
                log = ''
                for i, k in enumerate(self.topk):
                    log += 'Top {} Acc:{Acc.avg:2f}%\t'.format(k, Acc=topAcc[i])
                print(log)
                # print('------------Class Accuracy--------------')
                # for i, c in enumerate(self.classes):
                #     print('{} Acc:{Acc.avg:2f}\t'.format(c, Acc=classesAcc[i]))
                lr = iter(self.optimizer.param_groups).__next__()['lr']
                print('LearningRate:{}'.format(lr))

        print(
            'Training time (for an epoch) :{}'.format(
                time.time() -
                time_stamp))

        return topAcc, classesAcc, losses.avg

    def validate(self):
        self.model.eval()

        losses = AverageMeter()
        topAcc = [AverageMeter() for i in self.topk]
        classesAcc = [AverageMeter() for c in self.classes]

        time_stamp = time.time()
        for data, label in self.validate_loader:
            if self.cuda:
                data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, label)

            prec = self.accuracy(output.data, label.data, topk=self.topk)
            classes_count, classes_total = self.classes_accuracy(
                output.data, label.data)

            losses.updateWithVal(loss.data[0], data.size(0))
            for i, _ in enumerate(self.topk):
                topAcc[i].updateWithVal(prec[i], data.size(0))
            for i, _ in enumerate(self.classes):
                classesAcc[i].updateWithCount(
                    classes_count[i].item(),
                    classes_total[i].item())

        print('=======================================')
        current_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S')
        print('\n{}\t Validate_Avg_Loss: {loss.avg:.4f},\t'
              'Time: {:.4f}s\n'
              .format(
                  current_time,
                  time.time() - time_stamp,
                  loss=losses,
              ))

        print('------------Topk Accuracy--------------')
        log = ''
        for i, k in enumerate(self.topk):
            log += 'Top {} Acc:{Acc.avg:2f}\t'.format(k, Acc=topAcc[i])
        print(log)
        # print('------------Class Accuracy--------------')
        # for i, c in enumerate(self.classes):
        #     print('{} Acc:{Acc.avg:2f}\t'.format(c, Acc=classesAcc[i]))

        return topAcc, classesAcc, losses.avg

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
            train_topk_acc, train_class_acc, train_loss = self.train(e)

            bn_idx = 0
            for m in self.model.modules():
                if isinstance(
                        m, nn.BatchNorm2d) or isinstance(
                        m, nn.BatchNorm1d):
                    self.writer.add_histogram(
                        'bn'+str(bn_idx) + 'gamma', m.weight.data.cpu().numpy(), e)
                    self.writer.add_histogram(
                        'bn'+str(bn_idx) + 'beta', m.bias.data.cpu().numpy(), e)

                    bn_idx += 1

            validate_topk_acc, validate_class_acc, validate_loss = self.validate()

            # self.writer.add_scalar('/data/entropy',self.get_all_gamma_entropy(),e)
            LOSS = {'train': train_loss, 'validate': validate_loss}
            trainTopk = {}
            validateTopk = {}
            trainClasses = {}
            validateClasses = {}
            for i, k in enumerate(self.topk):
                trainTopk['Top%s' % k] = train_topk_acc[i].avg
                validateTopk['Top%s' % k] = validate_topk_acc[i].avg
            for i, c in enumerate(self.classes):
                trainClasses['%s' % c] = train_class_acc[i].avg
                validateClasses['%s' % c] = validate_class_acc[i].avg

            self.writer.add_scalars('Loss', LOSS, e)
            self.writer.add_scalars('Topk_Accuracy/Training', trainTopk, e)
            self.writer.add_scalars('Topk_Accuracy/Validation', validateTopk, e)
            self.writer.add_scalars(
                'Classes_Accuracy/Training', trainClasses, e)
            self.writer.add_scalars(
                'Classes_Accuracy/Validation', validateClasses, e)

            is_best = validate_topk_acc[0].avg > best_precision
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

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def classes_accuracy(self, output, target):

        classes_total = torch.zeros(len(self.classes))
        classes_count = torch.zeros(len(self.classes))

        target_idx = target.view(-1).to(torch.int)
        for idx in target_idx:
            classes_total[idx] += 1

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        classes_idx = pred.masked_select(correct).to(torch.int)

        for idx in classes_idx:
            classes_count[idx] += 1

       #  avoid_zero_classes_total = classes_total.clone()
       #  azct_np = avoid_zero_classes_total.numpy()
       #  azct_np[azct_np == 0] = 1

       #  res = torch.mul(classes_count, 100).div_(avoid_zero_classes_total)

        return classes_count.to(torch.int), classes_total.to(torch.int)

    def adjust_lr_cosine(self, epoch, batch, nBatch):
        T_total = self.epochs * nBatch
        T_cur = (epoch % self.epochs) * nBatch + batch
        lr = 0.5 * self.lr * (1 + math.cos(math.pi * T_cur / T_total))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class Network_Slimming_Trainer(Trainer):

    def __init__(self, **kwargs):
        super(Network_Slimming_Trainer, self).__init__(**kwargs)
        self.penalty = kwargs['penalty']
        self.root = os.path.join(self.root, 'lasso')
        #self.clipper = VaryingBNlrClipper(self.penalty)
        #self.layers_memory_cost = self.cal_memory_cost_()
        #self.BN_layers_y, self.BN_layers_xi = self.init_y_xi()
        # print(self.layers_memory_cost)
        # self.const_div = 2 / sqrt(pi)
        # self.const_norm = stats.norm(loc=0, scale=1 / sqrt(2))

    def train(self, e):
        self.model.train()

        losses = AverageMeter()
        topAcc = [AverageMeter() for i in self.topk]
        classesAcc = [AverageMeter() for c in self.classes]
        time_stamp = time.time()

        for batch_idx, (data, label) in enumerate(self.train_loader):
            # lr = self.adjust_lr_cosine(e,batch_idx,len(self.train_loader))

            if self.cuda:
                data, label = data.cuda(), label.cuda(async=True)

            output = self.model(data)
            loss = self.criterion(output, label)
            prec = self.accuracy(output.data, label.data, topk=self.topk)
            classes_count, classes_total = self.classes_accuracy(
                output.data, label.data)

            losses.updateWithVal(loss.data[0], data.size(0))
            for i, _ in enumerate(self.topk):
                topAcc[i].updateWithVal(prec[i], data.size(0))
            for i, _ in enumerate(self.classes):
                classesAcc[i].updateWithCount(
                    classes_count[i].item(),
                    classes_total[i].item())

            self.optimizer.zero_grad()
            loss.backward()
            self.updateBN()
            self.optimizer.step()

            if (batch_idx + 1) % self.log_interval == 0:
                current_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S')
                print('=======================================')
                print('{}\t Epoch: {} [{}/{}]\t'
                      'Loss: {loss.avg:.6f}\t'
                      .format(
                        current_time,
                        e,
                        (batch_idx + 1) * len(data),
                        len(self.train_loader.dataset),
                        loss=losses,
                          ))
                print('------------Topk Accuracy--------------')
                log = ''
                for i, k in enumerate(self.topk):
                    log += 'Top {} Acc:{Acc.avg:2f}%\t'.format(k, Acc=topAcc[i])
                print(log)
                # print('------------Class Accuracy--------------')
                # for i, c in enumerate(self.classes):
                #     print('{} Acc:{Acc.avg:2f}\t'.format(c, Acc=classesAcc[i]))
                lr = iter(self.optimizer.param_groups).__next__()['lr']
                print('LearningRate:{}'.format(lr))

        print(
            'Training time (for an epoch) :{}'.format(
                time.time() -
                time_stamp))

        return topAcc, classesAcc, losses.avg

    def updateBN(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(self.penalty*torch.sign(m.weight.data))
                # m.bias.grad.data.add_(self.penalty*torch.sign(m.bias.data))

    def l1_penalty(self):
        bn_l1_norm_all = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_l1_norm_all += m.weight.norm(
                    1) * m.lr
                print('lr:{}\n'.format(m.lr.data))
        return bn_l1_norm_all

    def MultiStepLR(self, e):
        if e in [self.epochs * 0.5, self.epochs * 0.75]:
            self.lr = self.lr * 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def quad_express(self, t, e):
        return self.const_div * pow(e, -t**2)

    # nb_quad_express = cfunc("float32(float32)")(quad_express)

    # @vectorize(["float32(float32)"],target="cuda")

    def grad_cal(self, x):
        x = x / 0.01
        erf = quad(self.quad_express, 0, x)[0]
        cdf = 2 * self.const_norm.cdf(x) * x
        return erf + cdf

    def DLASSO_update(self):
        vectorized_cal = vectorize(self.grad_cal)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(
                    self.penalty *
                    torch.Tensor(
                        vectorized_cal(
                            m.weight.data)).cuda())

    def init_y_xi(self):
        BN_layers_y = []
        BN_layers_xi = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                BN_layers_y.append(torch.zeros(m.weight.data.size()))
                BN_layers_xi.append(0)
        return BN_layers_y, BN_layers_xi

    def Fast_ISTA_update(self, lr):
        idx = 0

        # print(gamma_k)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                xi_k_1 = self.BN_layers_xi[idx]
                xi_k = (1 + sqrt(1 + 4 * xi_k_1**2)) / 2
                gamma_k = (1 - xi_k_1) / xi_k
                self.BN_layers_xi[idx] = xi_k
                penalty_factors = m.weight.data.abs().sub(
                    self.layers_memory_cost[idx] * self.penalty * lr)
                penalty_factors.masked_fill_(penalty_factors.lt(0), 0)
                y = penalty_factors.mul(torch.sign(m.weight.data))
                m.weight.data.copy_(y)
                weight_k_1 = m.weight.data
                weight_k = (
                    (1 - gamma_k) * y + gamma_k * self.BN_layers_y[idx])
                if weight_k_1.norm(1) < weight_k.norm(1):
                    m.weight.data.copy_(y)
                else:
                    m.weight.data.copy_(weight_k)
                self.BN_layers_y[idx].copy_(y)
                idx += 1

    def cal_memory_cost_(self):
        layers_memory_cost = []
        conv_layer_input_size = []
        conv_layer_ouput_size = []
        conv_layer_pre_kernel_size = []
        conv_layer_post_kernel_size = []
        handle_list = []

        def conv_hook(self, input, output):
            batch_size, in_channels, in_height, in_width = input[0].size()
            out_channels, out_height, out_width = output[0].size()
            conv_layer_input_size.append(in_height * in_width)
            conv_layer_ouput_size.append(out_height * out_width)
            conv_layer_pre_kernel_size.append(
                self.kernel_size[0] * self.kernel_size[1] * in_channels)
            conv_layer_post_kernel_size.append(
                self.kernel_size[0] * self.kernel_size[1] * out_channels)

        def fc_hook(self, input, output):
            out_features, in_features = self.weight.size()
            conv_layer_input_size.append(in_features)
            conv_layer_ouput_size.append(out_features)
            conv_layer_pre_kernel_size.append(in_features)
            conv_layer_post_kernel_size.append(out_features)

        def init_hooks(model):
            children = list(model.children())
            if not children:
                if isinstance(model, nn.Conv2d):
                    handle_list.append(model.register_forward_hook(conv_hook))
                if isinstance(model, nn.Linear):
                    handle_list.append(model.register_forward_hook(fc_hook))
            for c in children:
                init_hooks(c)

        input_data = Variable(
            torch.rand(3, 32, 32).unsqueeze(0),
            requires_grad=True)
        init_hooks(self.model)
        self.model.eval()
        self.model(input_data)

        for idx, pre_size in enumerate(conv_layer_pre_kernel_size[:-1]):
            subsequent_layers_cost = sum([post_size
                                          for post_size in conv_layer_post_kernel_size
                                          [idx + 1:]]) + sum([output_size
                                                              for output_size in conv_layer_ouput_size[idx + 1:]])
            layers_memory_cost.append(
                (subsequent_layers_cost + pre_size) /
                conv_layer_input_size[idx])

        for h in handle_list:
            h.remove()
        max_item_log = math.log(max(layers_memory_cost), 10)
        for i in range(len(layers_memory_cost)):
            layers_memory_cost[i] = math.log(
                layers_memory_cost[i], 10) / max_item_log

        return layers_memory_cost

    def scale_BN_Conv(self, scale_factor):
        pre_layer_is_BN = False
        for m in self.model.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(
                    m, nn.Linear)) and pre_layer_is_BN:
                m.weight.data.mul_(1 / scale_factor)
                pre_layer_is_BN = False
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.mul_(scale_factor)
                pre_layer_is_BN = True

    def regularization_loss(self):
        idx = 0
        total_loss = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                total_loss += m.weight.data.norm(1)
                idx += 1

        return total_loss


class KD_FineTuning_Trainer(Trainer):

    def __init__(self, **kwargs):
        super(KD_FineTuning_Trainer, self).__init__(**kwargs)
        self.teacher_model = kwargs['teacher']
        self.transfer_criterion = kwargs['transfer_criterion']
        self.loss_ratio = kwargs['loss_ratio']
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    # def KnowledgeDistillation(self):
    #    KD_Loss = []
    #    activationMaps_student = []
    #    activationMaps_teacher = []

    #    def activationMap_student_hook(self,input,output):
    #        activationMaps_student.append(output.mean(dim=0, keepdim=True))

    #    def activationMap_teacher_hook(self,input,output):
    #        activationMaps_teacher.append(output.mean(dim=0, keepdim=True))

    #    def init_hook(model, hook):
    #        children = list(model.children())
    #        if not children:
    #            if isinstance(model, nn.Conv2d):
    #                model.register_forward_hook(hook)
    #        for i in children:
    #            init_hook(i, hook)
    #    init_hook(self.model, activationMap_student_hook)
    #    init_hook(self.teacher_model, activationMap_teacher_hook)
    #    for i,j in zip(activationMap_student_hook, activationMap_teacher_hook):
    #        KD_Loss.append()
    def train(self, e):
        self.model.train()

        losses = AverageMeter()
        topAcc = [AverageMeter() for i in self.topk]
        classesAcc = [AverageMeter() for c in self.classes]
        time_stamp = time.time()

        for batch_idx, (data, label) in enumerate(self.train_loader):
            # lr = self.adjust_lr_cosine(e,batch_idx,len(self.train_loader))

            if self.cuda:
                data, label = data.cuda(), label.cuda(async=True)

            output = self.model(data)
            teacher_output = self.teacher_model(data)
            transfer_loss = self.transfer_criterion(output, teacher_output)
            loss = (1 - self.loss_ratio) * self.criterion(output,
                                                          label) + self.loss_ratio * transfer_loss

            prec = self.accuracy(output.data, label.data, topk=self.topk)
            classes_count, classes_total, res = self.classes_accuracy(
                output[0].data, label.data)

            losses.update(loss.data[0], data.size(0))
            for i, _ in enumerate(self.topk):
                topAcc[i].update(prec[i], data.size(0))
            for i, _ in enumerate(self.classes):
                classesAcc[i].update(
                    res[i].item(),
                    classes_total[i].item(),
                    classes_count[i].item())

            self.optimizer.zero_grad()
            loss.backward()
            self.update()
            self.optimizer.step()

            if (batch_idx + 1) % self.log_interval == 0:
                print('Epoch: {} [{}/{}]\t'
                      'Loss: {loss.avg:.6f}\t'
                      .format(
                        e,
                        (batch_idx + 1) * len(data),
                        len(self.train_loader.dataset),
                        loss=losses,
                          ))
                for i, k in enumerate(self.topk):
                    print('Top{}Acc:{Acc.avg:2f}\t'.format(k, Acc=topAcc[i]))
                for i, c in enumerate(self.classes):
                    print('{} Acc:{Acc.avg:2f}\t'.format(c, Acc=classesAcc[i]))
                lr = iter(self.optimizer.param_groups).__next__()['lr']
                print('LearningRate:{}'.format(lr))

        print(
            'Training time (for an epoch) :{}'.format(
                time.time() -
                time_stamp))

        return topAcc, classesAcc, losses.avg


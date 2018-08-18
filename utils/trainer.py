import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
import os
import time
import math
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
    ]


class VaryingBNlrClipper(object):

    def __init__(self, penalty_range):
        self.penalty_range = penalty_range

    def __call__(self, module):
        if hasattr(module, 'lr'):
            w = module.lr.data
            module.lr.data = torch.sigmoid(w) * self.penalty_range


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
        # self.scheduler = lr_scheduler.MultiStepLR(
        #    self.optimizer, milestones=[
        #        self.epochs * 0.5, self.epochs * 0.75], gamma=0.1)
        # lamba1 = lambda epoch: 0.97 ** epoch
        # self.scheduler = lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     lr_lambda=lamba1
        # )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.epochs)

    def train(self, e):
        self.model.train()
        for p in self.plugins['iteration']:
            p.reset()

        time_stamp = time.time()

        for batch_idx, (data, label) in enumerate(self.train_loader):
            # lr = self.adjust_lr_cosine(e,batch_idx,len(self.train_loader))

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

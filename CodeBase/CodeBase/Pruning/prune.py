# coding=utf-8
# prune.py
import os
import math
import torch
from torch import nn
from CodeBase.Models import *
from CodeBase.Utils import *
from CodeBase.Pruning.measureCompressionAndAcceleration import measure_model


__all__ = ['pruneVGG16']


def pruneVGG16(trainer, checkpoint, save_path, ratio=0.5):
    trainer.load_checkpoint(checkpoint)
    print('\nPruning Start\n')
    cfg, cfg_mask = preprocess(trainer, ratio)
    pruned_model = transfer(trainer, save_path, cfg, cfg_mask)
    measure_model(trainer.model, pruned_model)
    return pruned_model


def preprocess(trainer, ratio):
    print("Before pre-processing the validation result is")
    trainer.validate()
    original_model = trainer.model
    total = 0
    bn = []
    # bias = []

    def flatten(l): return [item for sublist in l for item in sublist]

    #  count total channels & sort the gammas of each bn
    for m in original_model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            total += m.weight.data.shape[0]
            bn.append(m.weight.data.abs())

    bn = sorted(flatten(bn))
    # bias = flatten(bias)
    # bias.sort()
    threshold_idx = math.floor(total * ratio)
    threshold = bn[threshold_idx]
    print("Pruning Threshold: {}".format(threshold))

    pruned = 0
    cfg = []
    cfg_mask = []
    first_dropout = False
    remains = 0
    i = 0
    # set thoes gamma which are less than threshold to zero
    # handle the feature layers individually
    for m in original_model.conv_layers:
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.clone()
            normalization = weight_copy.data.abs()
            mask = normalization.ge(threshold).float().cuda()
            remains = int(torch.sum(mask))
            num_channels_at_least = round(weight_copy.size(0) * 0.05)

            if remains < num_channels_at_least:
                remains = num_channels_at_least
                bn, bn_sorted_idx = torch.sort(normalization)
                mask[bn_sorted_idx[-num_channels_at_least:]] = 1
            pruned += mask.shape[0] - remains
            m.weight.data.mul_(mask)
            # m.bias.data.mul_(mask)
            cfg.append(remains)
            cfg_mask.append(mask.clone())

            print(
                'Layer_idx: {:d} \t Total_channels: {:d} \t Remained_channels: {:d}'.format(
                    i, mask.shape[0], remains))

            i += 1

        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

        elif isinstance(m, nn.Dropout):
            if not first_dropout:
                cfg.append('D1')
                first_dropout = True

            else:
                cfg.append('D')

    # handle the classifier layers individually
    for m in original_model.classifier:
        if isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.clone()
            normalization = weight_copy.data.abs()
            mask = normalization.ge(threshold).float().cuda()
            remains = int(torch.sum(mask))
            num_channels_at_least = round(weight_copy.size(0) * 0.05)

            if remains < num_channels_at_least:
                remains = num_channels_at_least
                bn, bn_sorted_idx = torch.sort(normalization)
                mask[bn_sorted_idx[-num_channels_at_least:].data] = 1

            pruned += mask.shape[0] - remains
            m.weight.data.mul_(mask)
            # m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

            print(
                'Layer_idx: {:d} \t Total_channels: {:d} \t Remained_channels: {:d}'.format(
                    i, mask.shape[0], remains))

            i += 1

    pruned_ratio = pruned / total

    print("Pre-processing done! {}".format(pruned_ratio))
    print("After pre-processing the validation result:")
    trainer.validate()

    return cfg, cfg_mask


def transfer(trainer, save_path, cfg, cfg_mask):
    # transfer the reserved gamma to a newly built model
    original_model = trainer.model
    pruned_model = vgg_diy(
        num_classes=len(
            trainer.train_loader.dataset.classes),
         cfg=cfg)
    if trainer.cuda:
        pruned_model.cuda()

    layer_idx = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_idx]
    last_linear = False
    # first_linear = False
    discard_idx = None
    residual_bn_bias = None
    absorted_bias = None

    # transfer gamma from original model to pruned model
    for (m, m_new) in zip(original_model.modules(), pruned_model.modules()):
        idx0 = torch.squeeze(torch.nonzero(start_mask))
        idx1 = torch.squeeze(torch.nonzero(end_mask))

        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            discard_idx = torch.squeeze(torch.nonzero(end_mask.eq(0)))
            m_new.weight.data = m.weight.data[idx1].clone()
            m_new.bias.data = m.bias.data[idx1].clone()

            if discard_idx.size() == torch.Size(
                    [0]) or discard_idx.size() == torch.Size(
                    []):
                residual_bn_bias = None

            else:
                residual_bn_bias = m.bias.data[discard_idx].clone()

            m_new.running_mean = m.running_mean[idx1].clone()
            if absorted_bias is not None:
                m_new.running_mean.sub_(absorted_bias)

            m_new.running_var = m.running_var[idx1].clone()
            layer_idx += 1
            start_mask = end_mask.clone()

            if layer_idx < len(cfg_mask):
                end_mask = cfg_mask[layer_idx]

            else:
                last_linear = True

        elif isinstance(m, nn.Conv2d):
            w = m.weight.data[:, idx0.tolist(), :, :].clone()
            m_new.weight.data = w[idx1.tolist(), :, :, :].clone()
            m_new.bias.data = m.bias.data[idx1.tolist()].clone()

            if residual_bn_bias is not None:
                w = m.weight.data[:, discard_idx.tolist(), :, :]
                w = w[idx1.tolist(), :, :, :].clone()
                dim0, dim1 = w.size(0), w.size(1)

                w = w.view(dim0, dim1, -1).transpose(1, 2)
                absorted_bias = w.mul_(residual_bn_bias).sum(2).sum(1)

            else:
                absorted_bias = None
            #     m_new.bias.data.add_(absorted_bias)
            # print("idx0 {}\nidx1 {}\n".format(idx0.size(),idx1.size()))

        elif isinstance(m, nn.Linear):
            w = m.weight.data[:, idx0.tolist()].clone()

            if not last_linear:
                m_new.weight.data = w[idx1.tolist(), :].clone()
                m_new.bias.data = m.bias.data[idx1.tolist()].clone()
                if residual_bn_bias is not None:
                    w = m.weight.data[:, discard_idx.tolist()].clone()
                    w = w[idx1.tolist(), :]
                    absorted_bias = w.mul_(residual_bn_bias).sum(1)

                else:
                    absorted_bias = None
                # # print(absorted_bias)
                # m_new.bias.data.add_(absorted_bias)
            else:
                m_new.weight.data = w.clone()
                m_new.bias.data = m.bias.data.clone()
                if residual_bn_bias is not None:
                    w = m.weight.data[:, discard_idx.tolist()].clone()
                    absorted_bias = w.mul_(residual_bn_bias).sum(1)
                    m_new.bias.data.add_(absorted_bias)
                else:
                    absorted_bias = None

    print('Pruning done! Channel pruning result:{}'.format(cfg))
    torch.save({'cfg': cfg, 'model_state_dict': pruned_model.state_dict()},
               os.path.join(save_path, 'model_pruned.pkl'))

    trainer.model = pruned_model
    trainer.validate()
    trainer.model = original_model
    return pruned_model

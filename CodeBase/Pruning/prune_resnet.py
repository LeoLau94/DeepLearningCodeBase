# coding=utf-8
# prune_resnet.py
import math
import torch
from torch import nn
from CodeBase.Models import *
from CodeBase.Utils import *


__all__ = ['preprocessResNetPreActivation', 'transferResNetPreActivation']


def normalize(weight):
    max_weight = weight.max()
    min_weight = weight.min()
    return (weight - min_weight) / (max_weight - min_weight)


def preprocessResNetPreActivation(pruner, ratio):
    original_model = pruner.original_model
    total = 0
    count_bn = 0
    bn = []
    # bias = []

    def flatten(l): return [item for sublist in l for item in sublist]

    #  count total channels & sort the gammas of each bn
    for m in original_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            count_bn += 1

            if count_bn != 1:
                total += m.weight.data.size(0)
                # .mul_(m.weight.size(0))
                bn.append(normalize(m.weight.data.abs()))

            if count_bn == 3:
                count_bn = 0

    bn = sorted(flatten(bn))
    threshold_idx = math.floor(total * ratio)
    threshold = bn[threshold_idx]
    print("Pruning Threshold: {}".format(threshold))

    pruned = 0
    cfg = []
    cfg_mask = []

    remains = 0
    i = 0
    cfgs = []
    count_bn = 0

    for l in original_model.conv_layers:
        cfg = []
        for m in l.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn != 1:
                    normalization = normalize(m.weight.data.abs())
                    mask = normalization.gt(threshold).float().cuda()
                    remains = int(torch.sum(mask))
                    # print(mask)

                else:
                    mask = torch.ones(m.weight.size()).float().cuda()
                    remains = int(torch.sum(mask))

                pruned += mask.shape[0] - remains
                m.weight.data.mul_(mask)
                # m.bias.data.mul_(mask)
                cfg.append(remains)
                cfg_mask.append(mask.clone())

                print(
                    'layer_idx: {:d} \t total_channels: {:d} \t remained_channels: {:d}'.format(
                        i, mask.shape[0], remains))

                i += 1

                if count_bn == 3:
                    count_bn = 0

        cfgs.append(cfg)

    for j in range(len(cfgs)-1):
        cfgs[j].append(cfgs[j+1][0])

    remains = original_model.bn.weight.size(0)
    cfgs[-1].append(remains)
    cfg_mask.append(torch.ones(original_model.bn.weight.size()))

    print(
        'layer_idx: {:d} \t total_channels: {:d} \t remained_channels: {:d}'.format(
            i,
            remains,
            remains))

    pruned_ratio = pruned / total

    print("Pre-processing done! {}".format(pruned_ratio))

    return cfgs, cfg_mask


def transferResNetPreActivation(pruner, cfgs, cfg_mask):
    # transfer the reserved gamma to a newly built model
    original_model = pruner.original_model
    pruned_model = pruner.pruned_model = preactivation_resnet164(
        num_classes=len(pruner.classes),
        resolution=pruner.input_size[-1],
        cfgs=cfgs)

    layer_idx = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_idx]

    # process the first conv_layer
    pruned_model.conv1.weight.data = original_model.conv1.weight.data.clone()

    pre_is_BN = False
    discard_idx = None
    residual_bn_bias = None
    absorted_bias = None

    for (l, l_new) in zip(original_model.conv_layers, pruned_model.conv_layers):
        for (m, m_new) in zip(l.modules(), l_new.modules()):
            if isinstance(m, nn.BatchNorm2d):
                pre_is_BN = True
                # print('layer_idx:{}'.format(layer_idx))
                # print('start_mask: {}'.format(start_mask.nonezero().size()))
                # print('end_mask: {}'.format(end_mask.nonezero().size()))
                discard_idx = torch.nonzero(end_mask.eq(0))
                idx1 = torch.squeeze(torch.nonzero(end_mask), 1)

                m_new.weight.data = m.weight.data[idx1].clone()
                m_new.bias.data = m.bias.data[idx1].clone()

                if discard_idx.size() == torch.Size(
                    [0]) or discard_idx.size() == torch.Size(
                        []):
                    residual_bn_bias = None

                else:
                    discard_idx = torch.squeeze(discard_idx, 1)
                    residual_bn_bias = m.bias.data[discard_idx].clone()

                m_new.running_mean = m.running_mean[idx1].clone()
                m_new.running_var = m.running_var[idx1].clone()

                if absorted_bias is not None:
                    m_new.running_mean.sub_(absorted_bias)

                start_mask = end_mask
                layer_idx += 1

                if layer_idx < len(cfg_mask):
                    end_mask = cfg_mask[layer_idx]

                else:
                    pass

            elif isinstance(m, nn.Conv2d):
                if pre_is_BN:
                    idx0 = torch.squeeze(torch.nonzero(start_mask), 1)
                    idx1 = torch.squeeze(torch.nonzero(end_mask), 1)
                    w = m.weight.data[:, idx0.tolist(), :, :]
                    m_new.weight.data = w[idx1.tolist(), :, :, :].clone()
                    m_new.bias.data = m.bias.data[idx1.tolist()].clone()

                    if residual_bn_bias is not None:
                        w = m.weight.data[:, discard_idx.tolist(), :, :]
                        w = w[idx1.tolist(), :, :, :].clone()

                        w = w.view(w.size(0), w.size(1), -1)
                        absorted_bias = w.mul_(
                            residual_bn_bias.unsqueeze(1)).sum(2).sum(1)

                    else:
                        absorted_bias = None

                    pre_is_BN = False

                else:
                    m_new.weight.data = m.weight.data.clone()
                    # print(m_new,m_new.weight.size())

            else:
                pass

    pruned_model.bn.weight.data = original_model.bn.weight.data.clone()
    pruned_model.bn.bias.data = original_model.bn.bias.data.clone()
    pruned_model.bn.running_mean = original_model.bn.running_mean.clone().sub_(absorted_bias)
    pruned_model.bn.running_var = original_model.bn.running_var.clone()
    pruned_model.fc.weight.data = original_model.fc.weight.data.clone()
    pruned_model.fc.bias.data = original_model.fc.bias.data.clone()

import os
import argparse
import torch
from torch import nn
from nets.resnet_pre_activation import *
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from utils.trainer import AverageMeter
import math
from utils.measure import *
from collections import OrderedDict


parser = argparse.ArgumentParser(description='Network Slimming---Prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default:cifar10)'
                    )
parser.add_argument('--num-classes', type=int, default=10,
                    help='humber of classes'
                    )
parser.add_argument(
    '--validate-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='batch size of validation (default:1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable cuda training'
                    )
parser.add_argument('--prune-rate', type=float, default=0.5,
                    help='sparse rate (default:0.5)'
                    )
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model needed to be pruned'
                    )
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save pruned model (default:./)'
                    )
parser.add_argument(
    '--gpu-devices', type=str, default='0',
    help='decide which gpu devices to use.For exmaple:0,1')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
print('Using gpu devices:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.save:
    if not os.path.exists(args.save):
        os.makedirs(args.save)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447],
        std=[0.247, 0.243, 0.262])
    validate_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize
                         ])
                         ),
        batch_size=args.validate_batch_size, shuffle=False, **kwargs
    )
elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441],
        std=[0.267, 0.256, 0.276])
    # normalize = transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    validate_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              ])
                          ),
        batch_size=args.validate_batch_size, shuffle=False, **kwargs
        )


def validate(model, criterion):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    time_stamp = time.time()
    for data, label in validate_loader:
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)
        output = model(data)
        loss = criterion(output, label)

        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

    print('\n Validate_Avg_Loss: {loss.avg:.4f},\t'
          'Top1_Acc: {top1.avg:.2f}%\t'
          'Top5_Acc: {top5.avg:.2f}%\t'
          'Time: {:.4f}s\n'
          .format(
            time.time() - time_stamp,
            loss=losses,
            top1=top1,
            top5=top5,
              ))
    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
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


def cal_prob(weight):
    weight_sum = weight.sum()
    return weight.div(weight_sum)


def pre_process(m, threshold):
    normalization = m.weight.data.abs()
    mask = normalization.gt(threshold).float().cuda()
    remains = int(torch.sum(mask))
    num_channels_at_least = round(normalization.size(0) * 0.05)
    if remains < num_channels_at_least:
        remains = num_channels_at_least
        bn, bn_sorted_idx = torch.sort(normalization)
        mask[bn_sorted_idx[-num_channels_at_least:]] = 1
    return mask, remains


model = preactivation_resnet164(num_classes=args.num_classes, resolution=32)
if args.cuda:
    model.cuda()
state_dict = torch.load(args.model)
new_state_dict = OrderedDict()
for k,v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
criterion = nn.CrossEntropyLoss()
# print(model)


print('\nPruning Start\n')
total = 0
count_bn = 0
bn = []


def flatten(l): return [item for sublist in l for item in sublist]


for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        count_bn += 1
        if count_bn != 1:
            total += m.weight.data.size(0)
            bn.append(m.weight.data.abs())  # .mul_(m.weight.size(0))
        if count_bn == 3:
            count_bn = 0

bn = sorted(flatten(bn))
threshold_idx = math.floor(total * args.prune_rate)
threshold = bn[threshold_idx]
print("Pruning Threshold: {}".format(threshold))
# threshold = 0.1

pruned = 0
cfg = []
cfg_mask = []
first_dropout = False

remains = 0
i = 0
cfgs = []
count_bn = 0

for l in model.conv_layers:
    cfg = []
    for m in l.modules():
        if isinstance(m, nn.BatchNorm2d):
            count_bn += 1
            if count_bn != 1:
                mask, remains = pre_process(m, threshold)
                # print(mask)
            else:
                mask = torch.ones(m.weight.size()).float().cuda()
                remains = int(torch.sum(mask))
            pruned += mask.shape[0] - remains
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(remains)
            cfg_mask.append(mask.clone())
            print(
                'Layer_idx: {:d} \t Total_channels: {:d} \t Remained_channels: {:d}'.format(
                    i, mask.shape[0], remains))
            i += 1
            if count_bn == 3:
                count_bn = 0

    cfgs.append(cfg)

for j in range(len(cfgs)-1):
    cfgs[j].append(cfgs[j+1][0])
remains = model.bn.weight.size(0)
cfgs[-1].append(remains)
print(
    'Layer_idx: {:d} \t Total_channels: {:d} \t Remained_channels: {:d}'.format(
        i,
        remains,
        remains))


pruned_ratio = pruned / total

print("Pre-processing done! {}".format(pruned_ratio))
validate(model, criterion)
print(cfgs)
new_model = preactivation_resnet164(
    num_classes=args.num_classes,
    resolution=32,
    cfgs=cfgs)
print(new_model)
if args.cuda:
    new_model.cuda()

layer_idx = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_idx]

# process the first conv_layer
new_model.conv1.weight.data = model.conv1.weight.data.clone()

pre_is_BN = False
discard_idx = None
residual_bn_bias = None
absorted_bias = None

for (l, l_new) in zip(model.conv_layers, new_model.conv_layers):
    for (m, m_new) in zip(l.children(),
                          l_new.children()):
        if isinstance(m, nn.BatchNorm2d):
            pre_is_BN = True
            discard_idx = torch.squeeze(torch.nonzero(end_mask.eq(0)))
            idx1 = torch.squeeze(torch.nonzero(end_mask))

            m_new.weight.data = m.weight.data[idx1].clone()
            m_new.bias.data = m.bias.data[idx1].clone()

            if discard_idx.dim() > 0:
                residual_bn_bias = m.bias.data[discard_idx].clone()
            else:
                residual_bn_bias = None

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
                idx0 = torch.squeeze(torch.nonzero(start_mask))
                idx1 = torch.squeeze(torch.nonzero(end_mask))
                w = m.weight.data[:, idx0.tolist(), :, :]
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

                pre_is_BN = False
            else:
                m_new.weight.data = m.weight.data.clone()
                # print(m_new,m_new.weight.size())
        else:
            pass
new_model.bn.weight.data = model.bn.weight.data.clone()
new_model.bn.bias.data = model.bn.bias.data.clone()
new_model.bn.running_mean = model.bn.running_mean.clone()
new_model.bn.running_var = model.bn.running_var.clone()
new_model.fc.weight.data = model.fc.weight.data.clone()
new_model.fc.bias.data = model.fc.bias.data.clone()


print('Pruning done! Channel pruning result:{}'.format(cfgs))
torch.save({'cfg': cfgs, 'model_state_dict': new_model.state_dict()},
           os.path.join(args.save, 'model_pruned.pkl'))
# # torch.save({'cfg': 'D', 'model_state_dict':model.state_dict()},os.path.join(args.save,'fake_pruned_model.pkl'))
validate(new_model, criterion)

model.cpu()
new_model.cpu()
print('Model before pruning:')
old_model_params = measure_layer_param(model)
old_model_flops = measure_layer_flops(model)

print('Model after pruning:')
new_model_params = measure_layer_param(new_model)
new_model_flops = measure_layer_flops(new_model)

print('Params_reduced:{:.2f}%\nFLOPs_reduced:{:.2f}%'.format(
    (old_model_params - new_model_params) * 100 / old_model_params,
    (old_model_flops - new_model_flops) * 100 / old_model_flops
    ))

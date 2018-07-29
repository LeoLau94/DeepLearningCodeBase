import os
import argparse
import torch
from torch import nn
from nets.my_vgg import vgg_diy
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from utils.trainer import AverageMeter
import math
from utils.measure import *
from utils.convert_DataParallel_Model import convert_DataParallel_Model_to_Common_Model

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
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

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
    weight_max = weight.max()
    return weight / weight_max


state_dict = torch.load(args.model)
#state_dict = convert_DataParallel_Model_to_Common_Model(state_dict)
# model = vgg_diy(num_classes=args.num_classes,cfgs=model_save['cfg'])
model = vgg_diy(num_classes=args.num_classes)
if args.cuda:
    model.cuda()
# model.load_state_dict(model_save['model_state_dict'])
model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()

print('\nPruning Start\n')

total = 0
bn = []
bias = []


def flatten(l): return [item for sublist in l for item in sublist]


for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        total += m.weight.data.shape[0]
        bn.append(m.weight.data.abs())


bn = flatten(bn)
bias = flatten(bias)
bn.sort()
bias.sort()
threshold_idx = math.floor(total * args.prune_rate)

threshold = bn[threshold_idx]
# threshold = 0.1
print("Pruning Threshold: {}".format(threshold))


pruned = 0
cfg = []
cfg_mask = []
first_dropout = False

remains = 0
i = 0
for m in model.conv_layers:
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
                i,
                mask.shape[0],
                remains))
        i += 1
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')
    elif isinstance(m, nn.Dropout):
        if not first_dropout:
            cfg.append('D1')
            first_dropout = True
        else:
            cfg.append('D')
for m in model.classifier:
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
                i,
                mask.shape[0],
                remains))
        i += 1

pruned_ratio = pruned / total

print("Pre-processing done! {}".format(pruned_ratio))
validate(model, criterion)

new_model = vgg_diy(num_classes=100, cfg=cfg)
if args.cuda:
    new_model.cuda()

layer_idx = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_idx]
last_linear = False
first_linear = False
discard_idx = None
residual_bn_bias = None
absorted_bias = None

for (m, m_new) in zip(model.modules(), new_model.modules()):
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
# print(new_model)
torch.save({'cfg': cfg, 'model_state_dict': new_model.state_dict()},
           os.path.join(args.save, 'model_pruned.pkl'))
# torch.save({'cfg': 'D', 'model_state_dict':model.state_dict()},os.path.join(args.save,'fake_pruned_model.pkl'))
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

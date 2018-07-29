import os
import argparse
import torch
from torchvision import datasets, transforms
from trainer import AverageMeter

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


kwargs = {'num_workers': 2, 'pin_memory': True}
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

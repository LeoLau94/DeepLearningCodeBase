import os
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from vgg import vgg
from time import time
import shutil

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')

parser.add_argument('--path', type=str, default='',
					help= 'test oriented model')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu-devices',type=str,default='0',help='decide which gpu devices to use.For exmaple:0,1')
parser.add_argument('--root',type=str,default='./', metavar='PATH', help='path to save checkpoint')
args = parser.parse_args()
 
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
     torch.cuda.manual_seed(args.seed)
     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
     

if args.root:
	if not os.path.exists(args.root):
		os.mkdir(args.root)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
     datasets.CIFAR10('../data',train=False,
          transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((.5,.5,.5),(.5,.5,.5))
               ])
          ),
     batch_size = 1,shuffle=True,**kwargs
     )


checkpoint = torch.load(args.path)
try:
     cfg = checkpoint['cfg']
except Exception as KeyError:
     cfg = None
model = vgg(cfg=cfg)
model.load_state_dict(checkpoint['model_state_dict'])

if args.cuda:
     model.cuda()
     print('Using gpu devices:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))  

criterion = nn.CrossEntropyLoss()

model.eval()
test_loss = 0
correct = 0
flag = False
criterion.size_average=False
total_time = 0
for data,label in test_loader:
     if args.cuda:
          data,label = data.cuda(),label.cuda()
     data,label = Variable(data,volatile=True),Variable(label)
     start_time = time()
     output = model(data)
     total_time += time() - start_time
     test_loss += criterion(output,label).data[0]
     pred = output.data.max(1,keepdim=True)[1]
     correct += pred.eq(label.data.view_as(pred)).cpu().sum()
test_loss /= len(test_loader.dataset)
print('\n Test_average_loss: {:.4f}\t Acc: {}/{} ({:.1f}%)\t Average_inference_time: {:.4f}s\n'.format(
     test_loss,
     correct,
     len(test_loader.dataset),
     100. * correct / len(test_loader.dataset),
     total_time / len(test_loader.dataset),
     ))
criterion.size_average=True
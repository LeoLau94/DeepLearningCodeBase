import os
import argparse
import torch
from torch import nn
import numpy as np
from nets.my_vgg import vgg_diy
from torchvision import datasets, transforms
from torch.autograd import Variable
from time import time
import shutil
parser = argparse.ArgumentParser(description='Network Slimming---Prune')
parser.add_argument('--dataset', type=str, default='cifar10',
	help='training dataset (default:cifar10)'
	)
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
	help='batch size of testing (default:1000)'
	)
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
parser.add_argument('--gpu-devices',type=str,default='0',help='decide which gpu devices to use.For exmaple:0,1')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
print('Using gpu devices:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.save:
	if not os.path.exists(args.save):
		os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
     validate_loader = torch.utils.data.DataLoader(
          datasets.CIFAR10('./data',train=False,
               transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                    ])
               ),
          batch_size = args.validate_batch_size,shuffle=True,**kwargs
          )
elif args.dataset == 'cifar100':
     validate_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100('./data',train=False,
               transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                    ])
               ),
          batch_size = args.validate_batch_size,shuffle=True,**kwargs
          )

def test():
	model.eval()
	test_loss = 0
	correct = 0
	flag = False
	criterion.size_average=False
	start_time = time()
	for data,label in validate_loader:
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data,label = Variable(data,volatile=True),Variable(label)
		output = model(data)
		test_loss += criterion(output,label).data[0]
		pred = output.data.max(1,keepdim=True)[1]
		correct += pred.eq(label.data.view_as(pred)).cpu().sum()
	test_loss /= len(validate_loader.dataset)
	print('\n Test_average_loss: {:.4f}\t Acc: {}/{} ({:.1f}%)\t Time: {:.4f}s\n'.format(
		test_loss,
		correct,
		len(validate_loader.dataset),
		100. * correct / len(validate_loader.dataset),
		time() - start_time,
		))
	criterion.size_average=True
	return correct / float(len(validate_loader.dataset))

model = vgg_diy()
if args.cuda:
	model.cuda()
model.load_state_dict(torch.load(args.model)['model_state_dict'])
criterion = nn.CrossEntropyLoss()

print('\nPruning Start\n')
total = 0
for m in model.modules():
	if isinstance(m,nn.BatchNorm2d):
		total += m.weight.data.shape[0]
bn = torch.zeros(total)
idx = 0
for m in model.modules():
	if isinstance(m,nn.BatchNorm2d):
		size = m.weight.data.shape[0]
		bn[idx:(idx+size)] = m.weight.data.abs().clone()
		idx += size
bn_sorted,bn_sorted_idx = torch.sort(bn)
threshold_idx = int(total * args.prune_rate)
threshold = bn_sorted[threshold_idx]
print("Pruning Threshold: {}".format(threshold))

pruned = 0
cfg = []
cfg_mask = []

for i,m in enumerate(model.modules()):
	if isinstance(m,nn.BatchNorm2d):
		weight_copy = m.weight.data.clone()
		mask = weight_copy.abs().gt(threshold).float().cuda()
		pruned += mask.shape[0] - torch.sum(mask)
		m.weight.data.mul_(mask)
		m.bias.data.mul_(mask)
		cfg.append(int(torch.sum(mask)))
		cfg_mask.append(mask.clone())
		print('Layer_idx: {:d} \t Total_channels: {:d} \t Remained_channels: {:d}'.format(
			i,mask.shape[0],int(torch.sum(mask))
			))
	elif isinstance(m,nn.MaxPool2d):
		cfg.append('M')

pruned_ratio = pruned / total

print("Pre-processing done! {}".format(pruned_ratio))
test()

new_model = vgg_diy(cfg=cfg)
if args.cuda:
	new_model.cuda()

layer_idx = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_idx]
change_first_linear = False

for ((i,m),m_new) in zip(enumerate(model.modules()),new_model.modules()):
	idx0 = torch.squeeze(torch.nonzero(start_mask))
	idx1 = torch.squeeze(torch.nonzero(end_mask))
	if isinstance(m,nn.BatchNorm2d):
		m_new.weight.data = m.weight.data[idx1].clone()
		m_new.bias.data = m.bias.data[idx1].clone()
		m_new.running_mean = m.running_mean[idx1].clone()
		m_new.running_var = m.running_var[idx1].clone()
		layer_idx += 1
		start_mask = end_mask.clone()
		if layer_idx < len(cfg_mask):
			end_mask = cfg_mask[layer_idx]
	elif isinstance(m,nn.Conv2d):
		w = m.weight.data[:,idx0.tolist(),:,:].clone()
		m_new.weight.data = w[idx1.tolist(),:,:,:].clone()
	elif isinstance(m,nn.Linear):
		if change_first_linear is False:
			m_new.weight.data = m.weight.data[:,idx0.tolist()].clone()
			change_first_linear = True
		else:
			pass

print('Pruning done! Channel pruning result:{}'.format(cfg))
torch.save({'cfg': cfg, 'model_state_dict':new_model.state_dict()},os.path.join(args.save,'model_pruned.pkl'))
model = new_model
test()
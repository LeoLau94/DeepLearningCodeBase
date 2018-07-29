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

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--fine-tune', default='', type=str, metavar='PATH',
                    help='fine-tune from pruned model')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu-devices',type=str,default='0',help='decide which gpu devices to use.For exmaple:0,1')
parser.add_argument('--root',type=str,default='./', metavar='PATH', help='path to save checkpoint')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
print('Using gpu devices:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
     torch.cuda.manual_seed(args.seed)

if args.root:
	if not os.path.exists(args.root):
		os.mkdir(args.root)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
     datasets.CIFAR10('../data',train=True,download=False,
          transform=transforms.Compose([
               transforms.Pad(4),
               transforms.RandomCrop(32),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((.5,.5,.5),(.5,.5,.5))
               ])
          ),batch_size=args.batch_size,shuffle=True,**kwargs
     )
test_loader = torch.utils.data.DataLoader(
     datasets.CIFAR10('../data',train=False,
          transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((.5,.5,.5),(.5,.5,.5))
               ])
          ),
     batch_size = args.test_batch_size,shuffle=True,**kwargs
     )

if args.fine_tune:
	checkpoint = torch.load(args.fine_tune)
	try:
		cfg = checkpoint['cfg']
	except Exception as KeyError:
		cfg = None
	model = vgg(cfg=cfg)
	model.load_state_dict(checkpoint['model_state_dict'])
else:
	cfg = None
	model = vgg(cfg=cfg)
if args.cuda:
	model.cuda()

optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()


def updateBN():
     for m in model.modules():
          if isinstance(m,nn.BatchNorm2d):
               m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))

def train(e):
	model.train()
	correct = 0
	train_size =0
	for batch_idx,(data,label) in enumerate(train_loader):
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data,label = Variable(data),Variable(label)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,label)
		loss.backward()
		if args.sr:
			updateBN()
		optimizer.step()
		pred = output.data.max(1,keepdim=True)[1]
		correct += pred.eq(label.data.view_as(pred)).cpu().sum()
		train_size += len(data)
		if (batch_idx + 1) % args.log_interval == 0:
			print("Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f}\t Acc: {:.6f}".format(
				e,
				(batch_idx + 1) * len(data),
				len(train_loader.dataset),
				100. * (batch_idx + 1) / len(train_loader),
				loss.data[0],
				correct / train_size
				))
			correct = 0
			train_size = 0

def test():
	model.eval()
	test_loss = 0
	correct = 0
	flag = False
	criterion.size_average=False
	start_time = time()
	for data,label in test_loader:
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data,label = Variable(data,volatile=True),Variable(label)
		output = model(data)
		test_loss += criterion(output,label).data[0]
		pred = output.data.max(1,keepdim=True)[1]
		correct += pred.eq(label.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\n Test_average_loss: {:.4f}\t Acc: {}/{} ({:.1f}%)\t Time: {:.4f}s\n'.format(
		test_loss,
		correct,
		len(test_loader.dataset),
		100. * correct / len(test_loader.dataset),
		time() - start_time,
		))
	criterion.size_average=True
	return correct / float(len(test_loader.dataset))

def save_checkpoint(state,is_best):
	file = os.path.join(args.root,'checkpoint.pkl')
	torch.save(state,file)
	if is_best:
		shutil.copyfile(file,os.path.join(args.root,'model_best.pkl'))


print(model)
print('\n-----Start Training-----\n')
best_precision = 0
for e in range(args.start_epoch,args.epochs):
	if e in [args.epochs*0.5, args.epochs*0.75]:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1
	train(e)
	precision = test()
	is_best = precision > best_precision
	training_state={
	'cfg': cfg,
	'start_epoch': e + 1,
	'model_state_dict': model.state_dict(),
	'optimizer': optimizer.state_dict(),
	'precision': precision,
	}
	save_checkpoint(
		training_state,
		is_best
		)
print("\n-----Training Completed-----\n")
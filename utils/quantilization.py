import torch
from torch import nn
from torch.autograd import Variable
from sklearn.cluster import KMeans
from torchvision import datasets,transforms


parser = argparse.ArgumentParser(description='Network Slimming---Prune')
parser.add_argument('--datasets', type=str, default='cifar10',
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = vgg()
if args.cuda:
	model.cuda()
model.load_state_dict(torch.load(args.model)['model_state_dict'])

def quantilized_params(m,m_hat):
	tmp = m.weight.data.clone().view(-1,1)
	res = KMeans.fit(tmp)
	m_hat.weight.data = torch.ByteTensor(res.labels_).view(m.weight.size())



for m in model.modules():
	
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.autograd import Variable
__all__ = ['SEBlock', 'SE_ResNet', 'se_resnet_18', 'se_resnet_34', 'se_resnet_50', 'se_resnet_101', 'se_resnet_152']



def conv3x3(in_channels,out_channels,stride=1):
	return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,
		padding=1,bias=False)


class SEBlock(nn.Module):
	def __init__(self,in_features,resolution):
		super(SEBlock,self).__init__()
		self.output=None
		self.globalAvgPool = nn.AvgPool2d(resolution,stride=1)
		self.fc1 = nn.Linear(in_features=in_features,out_features=round(in_features/16))
		self.fc2 = nn.Linear(in_features=round(in_features/16),out_features=in_features)

	def forward(self,x):
		out = self.globalAvgPool(x)
		out = out.view(out.size(0),-1)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
		out = F.sigmoid(out)
		self.output = out.clone()
		return out


class ResBlock(nn.Module):
	expansion = 1
	def __init__(self,in_channels,out_channels,stride=1,downsample=None,resolution=56):
		super(ResBlock,self).__init__()
		self.downsample = downsample
		self.conv_block = nn.Sequential(
			conv3x3(in_channels,out_channels,stride),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			conv3x3(out_channels,out_channels),
			nn.BatchNorm2d(out_channels),
			)
		self.seblock = SEBlock(out_channels*self.expansion,resolution)
		# self.seblock.register_forward_hook(get_SEBlock_output)

	def forward(self,x):
		residual = x if self.downsample is None else self.downsample(x)
		
		out = self.conv_block(x)

		original_out = out

		out = self.seblock(out)
		out = out.view(out.size(0),out.size(1),1,1)
		out = out * original_out
		out += residual

		return F.relu(out)


class Bottleneck(nn.Module):
	expansion = 4
	def __init__(self,in_channels,out_channels,stride=1,downsample=None,resolution=56):
		super(Bottleneck,self).__init__()
		self.downsample = downsample
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.Conv2d(out_channels,out_channels*4,kernel_size=1,bias=False),
			nn.BatchNorm2d(out_channels*4)
			)
		
		self.seblock = SEBlock(out_channels*self.expansion,resolution)
		# self.seblock.register_forward_hook(get_SEBlock_output)


	def forward(self,x):
		residual = x if self.downsample is None else self.downsample(x)

		out = self.conv_block(x)

		original_out = out
		out = self.seblock(out)
		out = out.view(out.size(0),out.size(1),1,1)
		out = out * original_out

		out += residual
		return F.relu(out)


class SE_ResNet(nn.Module):
	def __init__(self,block,layers,num_classes=1000,resolution=32):
		self.in_channels = 64
		super(SE_ResNet,self).__init__()

		self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

		self.resolution = round(resolution/4)
		
		self.layer1 = self._make_layer(block,64,layers[0])
		self.layer2 = self._make_layer(block,128,layers[1],stride=2)
		self.layer3 = self._make_layer(block,256,layers[2],stride=2)
		self.layer4 = self._make_layer(block,512,layers[3],stride=2)
		self.globalAvgPool = nn.AvgPool2d(kernel_size=self.resolution,stride=1)
		self.fc = nn.Linear(512*block.expansion,num_classes)

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0,math.sqrt(2./n))
			elif isinstance(m,nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self,block,out_channels,num_layers,stride=1):
		downsample = None
		if stride !=1 or self.in_channels != out_channels * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channels,out_channels*block.expansion,kernel_size=1,stride=stride,bias=False),
				nn.BatchNorm2d(out_channels*block.expansion),
				)
			self.resolution = round(self.resolution/stride)

		layers = []
		layers.append(block(self.in_channels,out_channels,stride,downsample,resolution=self.resolution))
		self.in_channels = out_channels*block.expansion
		for i in range(1,num_layers):
			layers.append(block(self.in_channels,out_channels,resolution=self.resolution))
		return nn.Sequential(*layers)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.maxpool(out)

		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)

		out = self.globalAvgPool(out)
		out = out.view(out.size(0),-1)
		out = self.fc(out)

		return out

def se_resnet_18(**kwargs):
	model = SE_ResNet(ResBlock,[2,2,2,2],**kwargs)
	return model

def se_resnet_34(**kwargs):
	model = SE_ResNet(ResBlock,[3,4,6,3],**kwargs)
	return model

def se_resnet_50(**kwargs):
	model = SE_ResNet(Bottleneck,[3,4,6,3],**kwargs)
	return model

def se_resnet_101(**kwargs):
	model = SE_ResNet(Bottleneck,[3,4,23,3],**kwargs)
	return model

def se_resnet_152(**kwargs):
	model = SE_ResNet(Bottleneck,[3,8,36,3],**kwargs)
	return model


if __name__ == '__main__':
	x = Variable(torch.randn((6,3,32,32))).cuda()
	net = se_resnet_34(num_classes=10,resolution=32)
	net = nn.DataParallel(net).cuda()
	# print(net)
	print(net(x).shape)
	for idx,m in enumerate(net.modules()):
		if isinstance(m,SEBlock):
			print(m.output)
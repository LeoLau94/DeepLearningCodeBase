import os
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from utils.trainer import Trainer,Network_Slimming_Trainer,Decrease_Entropy_Trainer,SE_Trainer
from nets.light_cnn import LightCNN_9Layers
from nets.deepid import *
from nets.net_sphere import *
from nets.my_vgg import vgg_diy
from nets.se_resnet import *
from utils.load_imglist import ImageList

parser = argparse.ArgumentParser(description='PyTorch fine-tuning')
parser.add_argument('--model', type=str, default='vgg',
                    help='model (default: vgg)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset (default: cifar10)')


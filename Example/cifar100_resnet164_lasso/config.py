import sys
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
sys.path.append('/data2/public/PyTorchCodeBase')
from CodeBase.Plugins import *
from CodeBase.Datasets import *
from CodeBase.Models import *


# all the variables must be list
__all__ = [
    'model',
    'optimizer',
    'train_loader',
    'validate_loader',
    'scheduler',
    'criterion',
    'plugins',
]


'''
GLOBAL KEY
'''
NUM_CLASSES = 100
BATCH_SIZE = 64
VAL_BATCH_SIZE = 1000
NUM_WORKERS = 2

LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LASSO_PENALTY = 0.0001

# initialize a model
model = preactivation_resnet164(num_classes=NUM_CLASSES)


# optimizers config
optimizer = optim.SGD(
   filter(
       lambda p: p.requires_grad,
       model.parameters()),
   lr=LR,
   weight_decay=WEIGHT_DECAY,
   momentum=MOMENTUM,
   nesterov=True)


# train_loaders & validate_loaders config
normalize = transforms.Normalize(
    mean=[0.507, 0.487, 0.441],
    std=[0.267, 0.256, 0.276])
dataset_root = '/data2/public/torchvision'
base_folder = 'cifar-100-python'
train_dataset = datasets.CIFAR100(dataset_root, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.ColorJitter(brightness=1),
                                     transforms.ToTensor(),
                                     normalize
                                     ]))
validate_dataset = datasets.CIFAR100(dataset_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                        ]))

train_dataset.classes = cifar_load_meta(
    dataset_root, base_folder, 'cifar100')
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True)
validate_loader = torch.utils.data.DataLoader(
     dataset=validate_dataset,
     batch_size=VAL_BATCH_SIZE,
     shuffle=False,
     num_workers=NUM_WORKERS,
     pin_memory=True
     )


# schedulers config
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 160)

# criterisons config
criterion = nn.CrossEntropyLoss()

# plugins config
plugins = []

plugins.append(LossMonitor())
plugins.append(TopKAccuracy(topk=(1, 5)))
plugins.append(IterationSummaryMonitor())
plugins.append(DistributionOfBNMonitor())
# plugins.append(ClassAccuracy())


def dataforward(self, data, target):
    output = self.trainer.model(data)
    loss = self.trainer.criterion(output, target)
    return output, loss
def updateBN(self):
    for m in self.model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.grad.data.add_(
                self.kwargs['penalty'] *
                torch.sign(
                    m.weight.data))

print('\nLASSO Training \n')
plugins.append(DataForward(dataforward))
plugins.append(ModelGradHandler(updateBN, penalty=LASSO_PENALTY))

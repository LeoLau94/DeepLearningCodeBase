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
NUM_CLASSES = 10
BATCH_SIZE = 64
VAL_BATCH_SIZE = 1000
NUM_WORKERS = 2

LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LOSS_RATIO =0.5

# initialize a model
model = vgg_diy(num_classes=NUM_CLASSES)

load_pkl = torch.load(
    '/data/leolau/checkpoints/CIFAR10/VGG16/Sep20_04-45-41/best_performance_model_params.pkl')
teacher_model = vgg_diy(num_classes=NUM_CLASSES)
teacher_model.load_state_dict(load_pkl)
teacher_model.cuda()
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False


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
    mean=[0.491, 0.482, 0.447],
    std=[0.247, 0.243, 0.262])
dataset_root = '/data2/public/torchvision'
base_folder = 'cifar-10-batches-py'
train_dataset = datasets.CIFAR10(dataset_root, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.ColorJitter(brightness=1),
                                     transforms.ToTensor(),
                                     normalize
                                     ]))
validate_dataset = datasets.CIFAR10(dataset_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                        ]))

train_dataset.classes = cifar_load_meta(
    dataset_root, base_folder, 'cifar10')
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
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 60)

# criterisons config
criterion = nn.CrossEntropyLoss()
transfer_criterion = nn.MSELoss()

# plugins config
plugins = []

plugins.append(LossMonitor(loggerName='Validation'))
plugins.append(TopKAccuracy(topk=(1, 5)))
plugins.append(IterationSummaryMonitor())
plugins.append(DistributionOfBNMonitor())
plugins.append(ClassAccuracy())


def dataforward(self, data, target):
    output = self.trainer.model(data)
    loss = self.trainer.criterion(output, target)
    return output, loss


def train_forward(self, data, target):
    student_output = self.trainer.model(data)
    teacher_output = self.kwargs['teacher_model'](data)

    transfer_loss = self.kwargs['transfer_criterion'](
        student_output, teacher_output)
    softmax_loss = self.trainer.criterion(student_output, target)
    loss = (1 - self.kwargs['loss_ratio']
            ) * softmax_loss + self.kwargs['loss_ratio'] * transfer_loss

    return student_output, loss

plugins.append(
    DataForward(
        train_forward,
        validation_forward=dataforward,
        teacher_model=teacher_model,
        loss_ratio=LOSS_RATIO,
        transfer_criterion=transfer_criterion))

print('\nPruning & Fine-Tuning with Knowledge Distillation\n')

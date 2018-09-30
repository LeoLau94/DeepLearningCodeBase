import sys
import torch.nn as nn
from torchvision import transforms, datasets
sys.path.append('/data2/public/PyTorchCodeBase')
from CodeBase.Plugins import *
from CodeBase.Datasets import *
from CodeBase.Models import *
from CodeBase.Trainer import *
from CodeBase.Pruning import *

# all the variables must be list
__all__ = [
    'model',
    'validate_loader',
    'criterion',
    'preprocess_method',
    'transfer_method',
    'plugins',
    'PRUNE_RATIO'
]


'''
GLOBAL KEY
'''
NUM_CLASSES = 100
VAL_BATCH_SIZE = 1000
NUM_WORKERS = 2

PRUNE_RATIO = 0.7
PARAMS = '/data/leolau/checkpoints/CIFAR100/ResNet164_preActivation(LASSO)/Sep20_05-01-30/best_performance_model_params.pkl'


# initialize a model
model = preactivation_resnet164(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(PARAMS))


# train_loaders & validate_loaders config
normalize = transforms.Normalize(
    mean=[0.507, 0.487, 0.441],
    std=[0.267, 0.256, 0.276])
dataset_root = '/data2/public/torchvision'
base_folder = 'cifar-100-python'
validate_dataset = datasets.CIFAR100(dataset_root, train=False,
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                        ]))
validate_dataset.classes = cifar_load_meta(
    dataset_root, base_folder, 'cifar100')
validate_loader = torch.utils.data.DataLoader(
     dataset=validate_dataset,
     batch_size=VAL_BATCH_SIZE,
     shuffle=False,
     num_workers=NUM_WORKERS,
     pin_memory=True
     )


# criterisons config
criterion = nn.CrossEntropyLoss()

#function config
preprocess_method = preprocessResNetPreActivation
transfer_method = transferResNetPreActivation

# plugins config
plugins = []

plugins.append(LossMonitor())
plugins.append(TopKAccuracy(topk=(1, 5)))
plugins.append(IterationSummaryMonitor())
# plugins.append(DistributionOfBNMonitor())
# plugins.append(ClassAccuracy())


def dataforward(self, data, target):
    output = self.trainer.model(data)
    loss = self.trainer.criterion(output, target)
    return output, loss

plugins.append(DataForward(dataforward))
print('\nPruning & Fine-Tuning with Knowledge Distillation\n')

import os
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils import *
from nets import *
from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')


# Training settings
parser = argparse.ArgumentParser(description='PyTorch training')

parser.add_argument('--model', type=str, default='vgg',
                    help='model (default: vgg)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset (default: cifar10)')
parser.add_argument(
    '--sparsity-regularization',
    '-sr',
    dest='sr',
    action='store_true',
    help='train with channel sparsity regularization')
parser.add_argument('-se', dest='se', action='store_true',
                    help='train with SEBlock')
parser.add_argument('--p', type=float, default=0.0001,
                    help='penalty (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--fine-tune', default='', type=str, metavar='PATH',
                    help='fine-tune from pruned model')
parser.add_argument(
    '--validate-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for validation (default: 1000)')
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
parser.add_argument(
    '--log-interval', type=int, default=100, metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--enable-class-accuracy',
    '-eca',
    dest='eca',
    action='store_true',
    default=False,
     help='enable class accuracy')
parser.add_argument(
    '--save-path',
    type=str,
    default='./save/',
    metavar='PATH',
    help='path to save checkpoint')
parser.add_argument('--num-workers', type=int, default=1,
                    help='how many threads to load data(default: 1)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--image-root-path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument(
    '--teacher_model', default=None, type=str, metavar='PATH',
    help='teacher model for knowledge distillation')
parser.add_argument(
    '--loss_ratio', default=0.2, type=float,
    help='ratio to control knowledge distillation\'s loss')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

model_dict = {
    'vgg': vgg_diy,
    'resnet': preactivation_resnet164,
    'sphereface': sphere20,
}


if args.model not in model_dict:
    raise ValueError('Name of network unknown %s' % args.model)
else:
    if args.fine_tune:
        load_pkl = torch.load(args.fine_tune)
        model = model_dict[args.model](
            num_classes=args.num_classes, cfg=load_pkl['cfg'])
        model.load_state_dict(load_pkl['model_state_dict'])
        if args.teacher_model is not None:
            teacher_model = model_dict[args.model](
                num_classes=args.num_classes)
            teacher_model.load_state_dict(torch.load(args.teacher_model))
        else:
            pass
        # model = model_dict[args.model](num_classes=args.num_classes)
        # model.load_state_dict(load_pkl)
        args.save_path = os.path.join(
            args.save_path,
            'fine_tune/' + args.model,
            args.dataset)
    else:
        model = model_dict[args.model](num_classes=args.num_classes)
        args.save_path = os.path.join(args.save_path, args.model, args.dataset)


kwargs = {'num_workers': args.num_workers,
          'pin_memory': True} if args.cuda else {}

dataset_root = '/data/torchvision/'
if args.dataset == 'cifar10':
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447],
        std=[0.247, 0.243, 0.262])

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
        dataset_root, base_folder, args.dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
     **kwargs)
    validate_loader = torch.utils.data.DataLoader(
         dataset=validate_dataset,
         batch_size=args.validate_batch_size, shuffle=False, **kwargs
         )

elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441],
        std=[0.267, 0.256, 0.276])

    base_folder = 'cifar-100-python'

    train_dataset = datasets.CIFAR100(
        dataset_root, train=True, download=True,
        transform=transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize]))
    validate_dataset = datasets.CIFAR100(dataset_root, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             normalize
                                             ]))

    train_dataset.classes = cifar_load_meta(
        dataset_root, base_folder, args.dataset)
    # normalize = transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
    # normalize = transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
     **kwargs)
    validate_loader = torch.utils.data.DataLoader(
         dataset=validate_dataset,
         batch_size=args.validate_batch_size, shuffle=False, **kwargs
         )

elif args.dataset == 'celeba':
    image_root = os.path.join(args.image_root_path, 'img_align_celeba')
    fileList = os.path.join(args.image_root_path, 'Anno/identity_CelebA.txt')
    attrLsit = os.path.join(args.image_root_path, 'Anno/list_attr_celeba.txt')
    loader = torch.utils.data.DataLoader(
        CelebADataset(root=image_root, fileList=fileList, attrLsit=attrLsit,),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

elif args.dataset == 'webface':
    train_root = os.path.join(args.image_root_path, 'webface_train')
    validate_root = os.path.join(args.image_root_path, 'webface_val')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=train_root,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(256, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ])
                             ),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    validate_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=validate_root,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        ),
        batch_size=args.validate_batch_size, shuffle=False, **kwargs)

else:
    pass

writer = SummaryWriter(log_dir=os.path.join(
    'runs', '|'.join([current_time, args.model, args.dataset])))


plugins = []
plugins.append(LossMonitor())
plugins.append(TopKAccuracy(topk=(1, 5)))
if args.eca:
    plugins.append(ClassAccuracy())


optimizer = optim.SGD(
   filter(
       lambda p: p.requires_grad,
       model.parameters()),
   lr=args.lr,
   weight_decay=args.weight_decay,
   momentum=args.momentum,
   nesterov=True)
# optimizer = optim.Adam(
#    filter(
#        lambda p: p.requires_grad,
#        model.parameters()),
#    lr=args.lr,
#    weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()


def dataforward(self, data, target):
    output = self.trainer.model(data)
    loss = self.trainer.criterion(output, target)
    return output, loss

if args.sr:
    print('\nTraining With LASSO\n')
    args.save_path = os.path.join(args.save_path, 'lasso')

    def updateBN(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(
                    self.kwargs['penalty'] *
                    torch.sign(
                        m.weight.data))
    plugins.append(DataForward(dataforward))
    plugins.append(ModelGradHandler(updateBN, penalty=args.p))

elif args.fine_tune is not None and args.teacher_model is not None:
    transfer_criterion = nn.MSELoss()
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print('\nTraining with Knowledge Distillation \n')

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
            loss_ratio=args.loss_ratio,
         transfer_loss=transfer_loss))

else:
    print('\nNormal Training \n')
    plugins.append(DataForward(dataforward))

trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr=args.lr,
        criterion=criterion,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        cuda=args.cuda,
        log_interval=args.log_interval,
        train_loader=train_loader,
        validate_loader=validate_loader,
        root=args.save_path,
        writer=writer,
        plugins=plugins

        )
trainer.start()

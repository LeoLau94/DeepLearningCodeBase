import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.trainer import *
from nets.my_vgg import vgg_diy
from nets.resnet_pre_activation import *
from nets.se_resnet import *
from utils.convert_DataParallel_Model import convert_DataParallel_Model_to_Common_Model
from command_parameter import *
from dataset_factory import dataset_factory


# parser from command_parameter
args = parser.parse_args()

# setting command parameters
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

# loading model and choose model
model_class = [vgg_diy,preactivation_resnet164,se_resnet_34]
model_dict  = dict(list(zip(model_name,model_class)))
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
    #model = model_dict[args.model](num_classes=args.num_classes)
    # model.load_state_dict(load_pkl)
    args.save_path = os.path.join(
        args.save_path,
        'fine_tune/' + args.model,
        args.dataset)
else:
    model = model_dict[args.model](num_classes=args.num_classes)
    args.save_path = os.path.join(args.save_path, args.model, args.dataset)


# dataset choice
kwargs = {'num_workers': args.num_workers,
          'pin_memory': True} if args.cuda else {}
'''
if args.dataset == 'cifar10':
    normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447],
        std=[0.247, 0.243, 0.262])
    train_loader = torch.utils.data.DataLoader(
         datasets.CIFAR10('./data', train=True, download=False,
                          transform=transforms.Compose([
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(),
                              #transforms.ColorJitter(brightness=1),
                              transforms.ToTensor(),
                              normalize
                              ])
                          ), batch_size=args.batch_size, shuffle=True, **kwargs
         )
    validate_loader = torch.utils.data.DataLoader(
         datasets.CIFAR10('./data', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              ])
                          ),
         batch_size=args.validate_batch_size, shuffle=False, **kwargs
         )
elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441],
        std=[0.267, 0.256, 0.276])
    # normalize = transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
    # normalize = transforms.Normalize((.5,.5,.5),(.5,.5,.5))
    train_loader = torch.utils.data.DataLoader(
         datasets.CIFAR100('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize
                               ])
                           ), batch_size=args.batch_size, shuffle=True, **kwargs
         )
    validate_loader = torch.utils.data.DataLoader(
         datasets.CIFAR100('./data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalize
                               ])
                           ),
         batch_size=args.validate_batch_size, shuffle=False, **kwargs
         )
else:
    train_loader = torch.utils.data.DataLoader(
         ImageList(root=args.image_root_path, fileList=args.image_train_list,
                   transform=transforms.Compose([
                       transforms.Resize(size=(args.img_size, args.img_size)),
                       transforms.RandomCrop(args.crop_size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                   ])
                   ),
         batch_size=args.batch_size, shuffle=True, **kwargs
         )
    validate_loader = torch.utils.data.DataLoader(
        ImageList(
            root=args.image_root_path, fileList=args.image_validate_list,
            transform=transforms.Compose(
                [transforms.Resize(
                     size=(args.crop_size, args.crop_size)),
                 transforms.ToTensor(), 
                 ])
            ),
        batch_size=args.validate_batch_size, shuffle=False, **kwargs
        )
'''

test_path       = '/home/leolau/pytorch/data' #just for test, when parameter confirm this might delete
dataset_f       = dataset_factory(args.dataset,args.batch_size,args.validate_batch_size,kwargs,root_path=test_path)
train_loader    = dataset_f.train_loader
validate_loader = dataset_f.validate_loader

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
transfer_criterion = nn.MSELoss()
if args.sr:
    print('\nSparsity Training \n')
    trainer = Network_Slimming_Trainer(
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
         penalty=args.p,
         )
elif args.se:
    print('\nSE_ResNet Training \n')
    trainer = SE_Trainer(
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
         SEBlock=SEBlock
         )

elif args.fine_tune is not None and args.teacher_model is not None: # other 3 00 01 10
    print('\nTraining with Knowledge Distillation \n')
    trainer = Trainer(
        model=model,
        teacher_model=teacher_model,
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
        loss_ratio=args.loss_ratio,
        transfer_criterion=transfer_criterion,

         )

else:
    print('\nNormal Training \n')
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

         )
trainer.start()

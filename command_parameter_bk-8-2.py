import argparse

# define of command parameters (may write as moudle)
parser = argparse.ArgumentParser(description='PyTorch training')

# model
model_name = ['vgg','resnet','se_resnet']
parser.add_argument('--model', type=str, default='vgg', choices=model_name,
                    help='model (default: vgg)')
# dataset
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset (default: cifar10)')
# sr
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
# se
parser.add_argument('-se', dest='se', action='store_true',
                    help='train with SEBlock')
# penalty
parser.add_argument('--p', type=float, default=0.0001,
                    help='penalty (default: 0.0001)')
# batch-size
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
# fine tune
parser.add_argument('--fine-tune', default='', type=str, metavar='PATH',
                    help='fine-tune from pruned model')
# validation batch size
parser.add_argument('--validate-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for validation (default: 1000)')
# epoch
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
# start-epoch
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# learning rate
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
# momentum
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# weight decay
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# resume
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# no cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# seed
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# log interval
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
# save path
parser.add_argument('--save-path', type=str, default='./save/', metavar='PATH',
                    help='path to save checkpoint')
# num workers(thread)
parser.add_argument('--num-workers', type=int, default=1,
                    help='how many thread to load data(default: 1)')
# num classes
parser.add_argument('--num_classes', type=int, default=10)

# images path
parser.add_argument('--image-root-path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
# images train list
parser.add_argument('--image-train-list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
# validation list
parser.add_argument('--image-validate-list',default='',type=str,metavar='PATH',
                    help='path to validation list (default: none)')
# image size
parser.add_argument('--img-size', '--img_size', default=144, type=int)

# crop size !!!crop size delete from command parameter,write into transfrom_config.xml
parser.add_argument('--crop-size', '--crop_size', default=128, type=int)

# teacher model
parser.add_argument('--teacher_model', default=None, type=str, metavar='PATH',
                    help='teacher model for knowledge distillation')
# loss ratio
parser.add_argument( '--loss_ratio', default=0.2, type=float,
                    help='ratio to control knowledge distillation\'s loss')
# end of define command parameters
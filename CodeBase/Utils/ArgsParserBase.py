import argparse


parser = argparse.ArgumentParser(
    description='PyTorch Training CodeBase',
    add_help=False,
    fromfile_prefix_chars='@',
)
parser.add_argument('--version', action='version', version='%(prog)s 0.1')

# args normal group settings
model = parser.add_argument_group('model')
dataset = parser.add_argument_group('dataset')
hyper_parameters = parser.add_argument_group('hyper parameters')
others = parser.add_argument_group('others')

# model settings
model.add_argument('--model', type=str, default='vgg',
                   help='model (default: vgg)')
model.add_argument(
    '--save-path',
    type=str,
    default='./save/',
    metavar='PATH',
    help='path to save checkpoint')

# dataset settings
dataset.add_argument('--dataset', type=str, default='cifar10',
                     help='dataset (default: cifar10)')

# hyperparameter settings
hyper_parameters.add_argument('--epochs', type=int, default=160, metavar='N',
                              help='number of epochs to train (default: 160)')
hyper_parameters.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')

# others settings
others.add_argument('--config', type=str, default='./config.py', metavar="PATH",
                    help='path to config.py')
others.add_argument(
    '--log-interval', type=int, default=100, metavar='N',
    help='how often does the trainer print the logs, the unit is batch.')
others.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
     help='how often does the trainer save the checkpoint, the unit is epoch')
others.add_argument(
    '--resume',
    type=str,
    default=None,
    help='checkpoint path to resume training')
others.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

Parser = parser
if __name__ == '__main__':
    print(parser.parse_args())

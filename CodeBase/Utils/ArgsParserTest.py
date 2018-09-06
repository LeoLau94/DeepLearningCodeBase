import argparse


parser = argparse.ArgumentParser(
    description='PyTorch Training CodeBase',
    add_help=False,
    prefix_chars='+-')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.add_argument('--config', type=str, default='./train.cfg',
                    metavar='Path',
                    help='training config path')

# args normal group settings
model = parser.add_argument_group('model')
others = parser.add_argument_group('others')
plugins = parser.add_argument_group('plugins')

# model settings
model.add_argument(
    '--model', type=str, default='vgg',
    help='model (default: vgg)')
model.add_argument(
    '--fine-tune', default='', type=str, metavar='PATH',
    help='fine-tune from pruned model')
model.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='path to latest checkpoint (default: none)')
model.add_argument(
    '--save-path',
    type=str,
    default='./save/',
    metavar='PATH',
    help='path to save checkpoint')

# plugin settings
plugins.add_argument(
    '--enable-class-accuracy',
    '-eca',
    dest='eca',
    action='store_true',
    default=False,
    help='enable class accuracy')

# others settings
others.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

if __name__ == '__main__':
    args = parser.parse_args()

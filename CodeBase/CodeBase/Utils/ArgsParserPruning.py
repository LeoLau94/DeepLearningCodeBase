import argparse
import ArgsParserBase as apb


parser = argparse.ArgumentParser(parents=[apb.parser])
# model = parser.model
model = parser.add_argument_group('model')
dataset = parser.add_argument_group('dataset')
hyper_parameters = parser.add_argument_group('hyper parameters')
others = parser.add_argument_group('others')
plugins = parser.add_argument_group('plugins')

model.add_argument(
    '--teacher_model', default=None, type=str, metavar='PATH',
    help='teacher model for knowledge distillation')
hyper_parameters.add_argument(
    '--loss_ratio', default=0.2, type=float,
    help='ratio to control knowledge distillation\'s loss')

others.add_argument(
    '--sparsity-regularization',
    '-sr',
    dest='sr',
    action='store_true',
    help='train with channel sparsity regularization')
others.add_argument('--p', type=float, default=0.0001,
                    help='penalty (default: 0.0001)')

if __name__ == '__main__':
    parser.parse_args()

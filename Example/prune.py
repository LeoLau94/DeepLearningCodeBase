# coding=utf-8
# prune.py
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from datetime import datetime
sys.path.append('/data2/public/PyTorchCodeBase')
from CodeBase.Utils import *
from CodeBase.Pruning import Pruner
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

# Arguments
args = Parser.parse_args()

# import config
config_dir, config_file = os.path.split(args.config)
config_file_name, _ = os.path.splitext(config_file)
sys.path.append(config_dir)
exec(' '.join(['from', config_file_name, 'import', '*']))
# decide using cuda or not
args.cuda = not args.no_cuda and torch.cuda.is_available()
# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# set benchmark
cudnn.benchmark = True
# set save path (to save checkpoint)
args.save_path = os.path.join(
    args.save_path,
    args.dataset,
    args.model,
    current_time)
# set writer
writer = SummaryWriter(log_dir=os.path.join(
    'runs', '[' + ']['.join([current_time, args.model, args.dataset]) + ']'))
# initialize Trianer
pruner = Pruner(
    model=model,
    cuda=args.cuda,
    root=args.save_path,
    validate_loader=validate_loader,
    criterion=criterion,
    preprocess_method=preprocess_method,
    transfer_method=transfer_method,
    plugins=plugins
)
pruner.prune(PRUNE_RATIO)

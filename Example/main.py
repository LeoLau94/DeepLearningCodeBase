import os
import sys
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from datetime import datetime
sys.path.append('/WorkSpace')
from CodeBase.Trainer import *
from CodeBase.Utils import *
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
    args.model,
    args.dataset,
    current_time)
# set writer
writer = SummaryWriter(log_dir=os.path.join(
    'runs', '|'.join([current_time, args.model, args.dataset])))
# initialize Trianer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    epochs=args.epochs,
    cuda=args.cuda,
    log_interval=args.log_interval,
    save_interval=args.save_interval,
    train_loader=train_loader,
    validate_loader=validate_loader,
    root=args.save_path,
    writer=writer,
    plugins=plugins
    )
if args.resume is not None:
    trainer.resume(args.resume)
trainer.start()

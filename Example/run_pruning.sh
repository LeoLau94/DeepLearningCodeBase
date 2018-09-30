ARGS='./cifar10_vgg16_prune/args.txt'
#ARGS='./cifar100_resnet164_prune/args.txt'
CONFIG='./cifar10_vgg16_prune/config_prune.py'
#CONFIG='./cifar100_resnet164_prune/config_prune.py'
CUDA_DEVICES=2
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python prune.py\
    --config=$CONFIG \
    @$ARGS

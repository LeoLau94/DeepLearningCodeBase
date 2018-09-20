ARGS='./cifar10_vgg16_prune/args.txt'
CONFIG='./cifar10_vgg16_prune/config.py'
CUDA_DEVICES=2
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python prune.py\
    --config=$CONFIG \
    @$ARGS \

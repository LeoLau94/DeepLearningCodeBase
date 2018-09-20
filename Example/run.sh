#ARGS='./webface_spherenet/args.txt'
#ARGS='./MsCeleb_spherenet/args.txt'
#ARGS='./cifar10_vgg16_lasso/args.txt'
#ARGS='./cifar10_vgg16/args.txt'
#ARGS='./cifar100_vgg16/args.txt'
#ARGS='./cifar100_vgg16_lasso/args.txt'
#ARGS='./cifar100_resnet164/args.txt'
ARGS='./cifar100_resnet164_lasso/args.txt'
#CONFIG='./webface_spherenet/config.py'
#CONFIG='./MsCeleb_spherenet/config.py'
#CONFIG='./cifar10_vgg16_lasso/config.py'
#CONFIG='./cifar10_vgg16/config.py'
#CONFIG='./cifar100_vgg16/config.py'
#CONFIG='./cifar100_vgg16_lasso/config.py'
#CONFIG='./cifar100_resnet164/config.py'
CONFIG='./cifar100_resnet164_lasso/config.py'
CUDA_DEVICES=6
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main.py\
    --config=$CONFIG \
    @$ARGS \



##########################
###   Configure path   ###
##########################

#MODEL_PATH=/home/leolau/pytorch/save/pruned/resnet/lasso/model_pruned.pkl
#MODEL_PATH=/home/leolau/pytorch/save/pruned/vgg/cifar10/lasso/model_pruned.pkl
#MODEL_PATH=/home/leolau/pytorch/save/pruned_naive/vgg/cifar10/lasso/model_pruned.pkl
#MODEL_PATH=/home/leolau/pytorch/save/pruned_naive/vgg/cifar100/lasso/model_pruned.pkl
MODEL_PATH=/home/leolau/pytorch/save/pruned/vgg/cifar100/model_pruned.pkl
#MODEL_PATH=/home/leolau/pytorch/save/vgg/cifar10/varying_BN/checkpoint.pkl
#MODEL_PATH=/home/leolau/pytorch/save/pruned/resnet/lasso/model_pruned.pkl
#TEACHER_MODEL_PATH=/home/leolau/pytorch/save/vgg/cifar10/best_precision_model_params.pkl
#TEACHER_MODEL_PATH=/home/leolau/pytorch/save/vgg/cifar100/best_precision_model_params.pkl
TEACHER_MODEL_PATH=/home/leolau/pytorch/save/vgg/cifar100/lasso/best_precision_model_params.pkl

############################
###   Image parameters   ###
############################



############################
###   Model parameters   ###
############################

NUM_CLASSES=100
MODEL=vgg
BATCH_SIZE=64
LR=0.01
EPOCH=60
WD=0.0001
NUM_WORKER=2
LOSS_RATIO=0.5

CUDA_VISIBLE_DEVICES=4 python main.py \
  --fine-tune=${MODEL_PATH} \
  --model=$MODEL  \
  --teacher_model=$TEACHER_MODEL_PATH \
  --num_classes=$NUM_CLASSES \
  --batch-size=$BATCH_SIZE \
  --lr=$LR \
  --epochs=$EPOCH\
  --wd=$WD\
  --num-worker=$NUM_WORKER\
  --dataset='cifar100'\
  --loss_ratio=$LOSS_RATIO\
  

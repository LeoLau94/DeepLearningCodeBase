

##########################
###   Configure path   ###
##########################

SAVE_PATH=/home/leolau/pytorch/save/


############################
###   Image parameters   ###
############################



############################
###   Model parameters   ###
############################

NUM_CLASSES=100
MODEL=vgg
BATCH_SIZE=64
LR=0.1
EPOCH=160
PENALTY=0.0001
WD=0.0001
NUM_WORKER=2

CUDA_VISIBLE_DEVICES=3 python main.py \
  --save-path=${SAVE_PATH} \
  --model=$MODEL  \
  --num_classes=$NUM_CLASSES \
  --batch-size=$BATCH_SIZE \
  --lr=$LR \
  --epochs=$EPOCH\
  --p=$PENALTY\
  --wd=$WD\
  --num-worker=$NUM_WORKER\
  -sr\
  --dataset='cifar100'

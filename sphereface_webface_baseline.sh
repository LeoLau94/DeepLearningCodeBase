

##########################
###   Configure path   ###
##########################

SAVE_PATH=/data/leolau/models/save/


############################
###   Image parameters   ###
############################

IMAGE_PATH=/data/WebFace

############################
###   Model parameters   ###
############################

NUM_CLASSES=10574
MODEL=sphereface
BATCH_SIZE=128
LR=0.05
EPOCH=60
WD=0.0001
NUM_WORKER=2

CUDA_VISIBLE_DEVICES=2 python main.py \
  --save-path=${SAVE_PATH} \
  --model=$MODEL  \
  --num_classes=$NUM_CLASSES \
  --batch-size=$BATCH_SIZE \
  --lr=$LR \
  --epochs=$EPOCH\
  --wd=$WD\
  --num-worker=$NUM_WORKER\
  --dataset='webface'\
  --image-root-path=${IMAGE_PATH}\
  --log-interval=10

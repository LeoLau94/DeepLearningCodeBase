MODEL_PATH=./save/resnet/cifar10/lasso/best_precision_model_params.pkl
SAVE_DIR=./save/pruned/resnet/lasso
PRUNE_RATE=0.5
python prune_resnet.py \
	--dataset='cifar10'\
   	--model=${MODEL_PATH}\
   	--save=${SAVE_DIR}\
   	--gpu-device=6\
	--prune-rate=$PRUNE_RATE\
	--num-classes=10

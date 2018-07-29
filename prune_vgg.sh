#MODEL_PATH=./save/vgg/cifar10/best_precision_model_params.pkl
#MODEL_PATH=./save/vgg/cifar100/best_precision_model_params.pkl
MODEL_PATH=./save/vgg/cifar100/lasso/best_precision_model_params.pkl
#SAVE_DIR=./save/pruned/vgg/cifar100/lasso
#SAVE_DIR=./save/pruned/vgg/cifar10/
SAVE_DIR=./save/pruned/vgg/cifar100/
PRUNE_RATE=0.6
python prune.py \
	--dataset='cifar100'\
   	--model=${MODEL_PATH}\
   	--save=${SAVE_DIR}\
   	--gpu-device=4\
	--prune-rate=$PRUNE_RATE\
	--num-classes=100

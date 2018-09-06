ARGS='./args.txt'
CONFIG='./config.py'
CUDA_DEVICES=4,5
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main.py\
    --config=$CONFIG \
    @$ARGS \

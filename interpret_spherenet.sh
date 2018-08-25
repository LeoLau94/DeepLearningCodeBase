MODEL="$HOME/data/models/save/sphereface/webface/Aug21_08-17-41/best_precision_model_params.pkl"
IMAGE_PATH="$HOME/data/CelebA"
SAVE_PATH="."

BATCH_SIZE=512

CUDA_VISIBLE_DEVICES=9 python make_interpretability.py\
    --model=$MODEL\
    --batch-size=$BATCH_SIZE\
    --num-workers=2 \
    --image-root-path=$IMAGE_PATH\
    --save-path=$SAVE_PATH \
    --no-inference


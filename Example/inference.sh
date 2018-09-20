MODEL='/data/leolau/checkpoints/MsCeleb/sphereface/Sep10_03-55-31/best_performance_model_params.pkl'
BATCH_SIZE=256
NUM_WORKERS=2
IMAGE_ROOT='/data2/public/CelebA_Face'
SAVE_PATH='./'
CUDA_VISIBLE_DEVICES=2 python make_interpretability.py \
    --model=$MODEL \
    --batch-size=$BATCH_SIZE \
    --num-workers=$NUM_WORKERS \
    --image-root-path=$IMAGE_ROOT \
    --save-path=$SAVE_PATH \
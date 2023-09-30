#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu "
    exit 1
fi
gpu="$1"
shift
set -e
    # --exclude_frame_index \
cmd="
python train.py \
    --manual_seed 42 \
    --net SCDepthV3 \
    --resnet_layers 18 \
    --dataset headcam_dataset \
    --skip_frames 1 \
    --pose_lr_mul 1 \
    --epoch 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --optim adam \
    --gpu "$gpu" \
    --tensorboard \
    --save_net 1 \
    --log_time \
    --html_logger \
    --workers 7 \
    --repeat 1 \
    --logdir './checkpoints/headcam/sequence/' \
    --suffix 'pose_lr_mul_{pose_lr_mul}_repeat_{repeat}_manual-seed_{manual_seed}' \
    --force_overwrite \
    --vis_every_vali 10 \
    --vis_every_train 10 \
    --vis_batches_vali 1 \
    --vis_batches_train 1 \
    $*"
echo $cmd
eval $cmd
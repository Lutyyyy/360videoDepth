#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu "
    exit 1
fi
gpu="$1"
shift
set -e
cmd="
python train.py \
    --manual_seed 42 \
    --net SCDepthV3 \
    --resnet_layers 18 \
    --dataset headcam_dataset \
    --use_frame_index \
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
    $*"
echo $cmd
eval $cmd
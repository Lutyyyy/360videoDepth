#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu "
    exit 1
fi
shift
set -e
cmd="
python train.py \
    --net SCDepthV3 \
    --resnet_layers 18 \
    --dataset tum \
    --skip_frames 10 \
    --pose_lr_mul 1 \
    --log_time \
    --epoch 100 \
    --batch_size 8 \
    --html_logger \
    --lr 1e-4 \
    --optim adam \
    --tensorboard \
    --gpu "$gpu" \
    --save_net 1 \
    --workers 4 \
    --repeat 1 \
    --logdir './checkpoints/tum/sequence/' \
    --suffix 'pose_lr_mul_{pose_lr_mul}_repeat_{repeat}' \
    --force_overwrite \
    $*"
echo $cmd
eval $cmd
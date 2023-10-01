#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu "
    exit 1
fi
gpu="$1"
shift
set -e
cmd="
python test.py \
    --manual_seed 42 \
    --net SCDepthV3 \
    --resnet_layers 18 \
    --dataset headcam_dataset \
    --output_dir "/home/usr/output0" \
    --suffix "pose_lr_mul_{pose_lr_mul}_repeat_{repeat}_manual-seed_{manual_seed}" \
    --checkpoint_path "/home/usr/ckptpath" \
    --workers 8 \
    --overwrite \
    --skip_frames 1 \
    --pose_lr_mul 1 \
    --epoch 80 \
    --batch_size 32 \
    --gpu "$gpu" \
    --html_logger \
    --repeat 1 \
    $*"
echo $cmd
eval $cmd

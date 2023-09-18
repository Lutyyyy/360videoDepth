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
    $*"
echo $cmd
eval $cmd
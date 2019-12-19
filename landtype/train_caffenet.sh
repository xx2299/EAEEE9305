#!/usr/bin/env sh
set -e

echo "begin"
./build/tools/caffe train \
    --solver=landtype/solver.prototxt \
     --weights=landtype/bvlc_alexnet.caffemodel
echo "end"

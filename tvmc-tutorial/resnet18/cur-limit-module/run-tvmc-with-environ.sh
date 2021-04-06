#!/bin/bash

# tvmc commands
TVMC="python3 -m tvm.driver.tvmc"
COMPILE="$TVMC compile"
TUNE="$TVMC tune"
RUN="$TVMC run"

# compile target
COMPILE_TARGET="llvm -mcpu=cascadelake"
COMPILE_TARGET_AVX512="llvm -mcpu=cascadelake-avx512"

# model path
resnet50="resnet50/resnet50-v2-7.onnx"
resnet18="resnet18/resnet18-v2-7.onnx"

NUM_THREADS="5"
export TVM_NUM_THREADS=$NUM_THREADS
echo "tune $resnet18 with $TVM_NUM_THREADS cores"
$TUNE --target "$COMPILE_TARGET" --output $resnet18-$NUM_THREADS.json $resnet18
echo "tune $resnet18 with $TVM_NUM_THREADS cores done"

NUM_THREADS="3"
export TVM_NUM_THREADS=$NUM_THREADS
echo "tune $resnet18 with $TVM_NUM_THREADS cores"
$TUNE --target "$COMPILE_TARGET" --output $resnet18-$NUM_THREADS.json $resnet18
echo "tune $resnet18 with $TVM_NUM_THREADS cores done"

NUM_THREADS="1"
export TVM_NUM_THREADS=$NUM_THREADS
echo "tune $resnet18 with $TVM_NUM_THREADS cores"
$TUNE --target "$COMPILE_TARGET" --output $resnet18-$NUM_THREADS.json $resnet18
echo "tune $resnet18 with $TVM_NUM_THREADS cores done"

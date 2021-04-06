#!/usr/bin/fish

# tvmc commands
set TVMC "python3 -m tvm.driver.tvmc"
set COMPILE "$TVMC compile"
set TUNE "$TVMC tune"
set RUN "$TVMC run"

set COMPILE_TARGET "llvm -mcpu=cascadelake"
set COMPILE_TARGET_AVX512 "llvm -mcpu=cascadelake-avx512"

# model path
set tvmcpath "/home/zl/tvm-tutorials/tvmc-tutorial/"
set resnet50 $tvmcpath"resnet50/resnet50-v2-7.onnx"
set resnet18 $tvmcpath"resnet18/resnet18-v2-7.onnx"

echo $resnet18

# use /usr/bin/time to measure time/cpu cost
set TIME "/usr/bin/time -v"

set num_threads 25 10 5 3 1

# tune and compile resnet18
for num in $num_threads
    set tunerecords $num-threads-resnet18.json
    set outputmodule $num-threads-resnet18.tar
    set time_env_prefix $TIME env TVM_NUM_THREADS=$num
    set tunecmd $time_env_prefix $TUNE --target \"$COMPILE_TARGET_AVX512\" --output $tunerecords $resnet18 
    set compilecmd $time_env_prefix $COMPILE --target \"$COMPILE_TARGET\" --tuning-records tunerecords --output outputmodule $resnet18 
    echo "Run tuning command: $tunecmd"
    eval $tunecmd
    echo "Run compile command: $compilecmd"
    eval $compilecmd
end



#!/bin/bash

COMMAND="/usr/bin/time -v \
    python3 -m tvm.driver.tvmc run \
    --inputs ../imagenet_cat.npz \
    --print-time \
    --repeat 200"

if [ -z "$1" ]; then
    echo "No current environment corenum info in parameter"
    echo "Run this script as: ./start-evaluation.sh [core-num]"
    exit 1
fi

# Get current docker core number from script parameter
corenum=$1
logfile_name=eval-logs/$corenum-cpu-execution-time.log

echo "Corenum: $corenum, logfile: $logfile_name"

# Iterate all tuned module, evaluate and save output to file
for cores in 60 40 20 15 10 5
do
    module_name="docker-"$cores"cpu-resnet152-module.tar"
    echo $module_name >> $logfile_name 
    
    EVAL_CMD="$COMMAND \
        $module_name \
        >> $logfile_name"

    echo "Evaluate tuned-module with $cores in container with $corenum cores."

    $EVAL_CMD

    echo "Evaluate tuned-module with $cores done."
done

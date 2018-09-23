#!/usr/bin/env bash

partition=AD2
gpus=8
job_name=$1

log_dir=logs/$job_name
rm $log_dir -rf
mkdir -p $log_dir

srun -u --partition=$partition --job-name=$job_name \
    --gres=gpu:$gpus -n1 --ntasks-per-node=1 \
    python3 $job_name.py $gpus $job_name \
    | tee $log_dir/log.txt

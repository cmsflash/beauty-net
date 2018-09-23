#!/usr/bin/env bash

gpus=1
job_name=$1

log_dir=logs/$job_name
rm $log_dir -rf
mkdir -p $log_dir

python3 $job_name.py $gpus $job_name \
    | tee $log_dir/log.txt


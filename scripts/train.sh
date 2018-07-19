#!/usr/bin/env bash

partition=AD1
gpus=1
job_name=test

srun -u --partition=${partition} --job-name=${job_name} \
    --gres=gpu:${gpus} -n1 --ntasks-per-node=1 \
    python3 train.py $gpus $job_name


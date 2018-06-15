#!/usr/bin/env bash

partition=local
gpus=1
job_name=local

python3 train.py $gpus $job_name

#!/usr/bin/env bash

partition=local
gpus=1
job_name=local
batch_size=$((1 * gpus))

python3 train.py \
  \
  --dataset="SCUT5500" \
  --data_dir="../scut-fbp5500/Images/" \
  --train_list="../scut-fbp5500/train_test_files/All_labels.txt" \
  --val_list="../scut-fbp5500/train_test_files/All_labels.txt" \
  \
  --network="BeautyNet" \
  --feature_extractor="MobileNetV2" \
  --classifier="Softmax" \
  --weight_decay=0.0005 \
  --loss="Cross Entropy" \
  \
  --input_height=320 \
  --input_width=320 \
  --train_resize_method="Data Augment" \
  --val_resize_method="Resize" \
  \
  --batch_size=$batch_size \
  --epochs=200 \
  \
  --optimizer="Adam" \
  --beta1=0.9 \
  --beta2=0.999 \
  \
  --lr_scheduler="Constant" \
  --lr=0.0001 \
  \
  --metrics "Accuracy" \
  --log_dir="logs/$job_name"

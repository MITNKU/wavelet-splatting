#!/bin/bash
data_path="./dataset/DIV2K_valid_LR_bicubic/X2"
iterations=50000
model_name="WIPESImage_Cholesky"
data_name="DIV2K_valid_LRX2"


if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 70000
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name $data_name --model_name $model_name --num_points $num_points --iterations $iterations --save_imgs
done
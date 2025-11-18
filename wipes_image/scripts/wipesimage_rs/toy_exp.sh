data_path="./dataset/toy_exp"
iterations=50000
model_name="WIPESImage_RS"
data_name="toy_exp"

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 1625
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name $data_name --model_name $model_name --num_points $num_points --iterations $iterations --save_imgs
done
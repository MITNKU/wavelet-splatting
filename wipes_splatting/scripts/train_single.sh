# Define global variables
export DATA_DIR=./3DGS_data/data                           # Base directory for data
export DATASET_NAME=nerf_synthetic                                                                   # Sub-dataset name
export SCENE_NAME=drums                                                                     # Scene name for the dataset
export DATA_PATH=$DATA_DIR/$DATASET_NAME/$SCENE_NAME                                        # Combine to get the full data path


# Define global variables for Model Path
export MODEL_DIR=./results                         # Model path
export DATE=$(date +"%Y%m%d%H%M")                                                           # Current date and time
export MODEL_FEATURE=original                                                               # Model feature
export MODEL_PATH=$MODEL_DIR/$DATASET_NAME/$SCENE_NAME/$MODEL_FEATURE/$DATE                 # Combine to get the full model path

# Define global variables for training settings
export PORT=21416                                                                           # Port number
export GPUID=0                                                                         # GPU ID to use

# Run the training script
CUDA_VISIBLE_DEVICES=$GPUID python train.py -s $DATA_PATH -m $MODEL_PATH --port $PORT --eval
CUDA_VISIBLE_DEVICES=$GPUID python render.py -m $MODEL_PATH --quiet
CUDA_VISIBLE_DEVICES=$GPUID python metrics.py -m $MODEL_PATH 




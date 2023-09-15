#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=05:00:00
#SBATCH --job-name=prep
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G


export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

root_dir="/scratch/s3412768/genre_NMT/en-$1"
# if root_dir does not exist, create it
if [ ! -d "$root_dir" ]; then
    mkdir -p $root_dir
fi

python /home1/s3412768/Genre-enabled-NMT/src/preprocess_genre.py \
    --lang $1 \
    --df $root_dir \

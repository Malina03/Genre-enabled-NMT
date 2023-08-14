#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=00:30:00
#SBATCH --job-name=pred
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G


export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

train_corpus=$1 # the corpus that the model was trained on
language=$2 # the target language
exp_type=$3 # type of experiment (genre-aware or baseline)
test_on=$4 # the test file to evaluate on, assuming it is placed in root_dir/data


root_dir="/scratch/s3412768/genre_NMT/en-${language}"

model="Helsinki-NLP/opus-mt-en-${language}"

test_file="${root_dir}/data/${test_on}"

checkpoint=$root_dir/models/$exp_type/$train_corpus/checkpoint-*

log_file="${root_dir}/logs/$exp_type/eval_${test_corpus}.log"
# if log directory does not exist, create it - but it really should exist
if [ ! -d "$root_dir/logs/$exp_type/" ]; then
    mkdir -p $root_dir/logs/$exp_type/
fi

    
python /home1/s3412768/Genre-enabled-NMT/src/train.py \
    --root_dir $root_dir \
    --train_file $test_file \
    --dev_file $test_file \
    --test_file $test_file\
    --gradient_accumulation_steps 2 \
    --batch_size 16 \
    --gradient_checkpointing \
    --adafactor \
    --exp_type $exp_type \
    --checkpoint $checkpoint \
    --model_name $model \
    --eval \
    --predict \
    &> $log_file 
    


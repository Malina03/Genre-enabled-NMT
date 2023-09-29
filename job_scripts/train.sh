#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=72:00:00
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G


export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

corpus=$1 # corpus to fine-tune on
language=$2 # target language
exp_type=$3 # type of experiment ([doc_]genre_aware[_token] -genres are added as proper tokens- or [doc_]baseline)
model_type=$4 # type of model (e.g. fine_tuned or from_scratch.)

root_dir="/scratch/s3412768/genre_NMT/en-$language"
log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$model_type/$exp_type/train_${corpus}.log"
# if log directory does not exist, create it
if [ ! -d "$root_dir/logs/$model_type/$exp_type" ]; then
    mkdir -p $root_dir/logs/$model_type/$exp_type
fi

model="Helsinki-NLP/opus-mt-en-${language}"



if [ $exp_type = 'genre_aware' ] || [ $exp_type = 'genre_aware_token' ]; then
    train_file="$root_dir/data/${corpus}.en-$language.train.tag.tsv"
    dev_file="${root_dir}/data/${corpus}.en-$language.dev.tag.tsv"
elif [ $exp_type = 'doc_genre_aware' ] || [ $exp_type = 'doc_genre_aware_token' ]; then
    train_file="$root_dir/data/${corpus}.en-$language.doc.train.tag.tsv"
    dev_file="${root_dir}/data/${corpus}.en-$language.doc.dev.tag.tsv"
elif [ $exp_type = 'baseline' ]; then
    train_file="$root_dir/data/${corpus}.en-$language.train.tsv"
    dev_file="${root_dir}/data/${corpus}.en-$language.dev.tsv"
elif [ $exp_type = 'doc_baseline' ]; then
    train_file="$root_dir/data/${corpus}.en-$language.doc.train.tsv"
    dev_file="${root_dir}/data/${corpus}.en-$language.doc.dev.tsv"
else   
    echo "Invalid experiment type"
    exit 1
fi

python /home1/s3412768/Genre-enabled-NMT/src/train.py \
    --root_dir $root_dir \
    --train_file $train_file \
    --dev_file $dev_file \
    --wandb \
    --gradient_accumulation_steps 2 \
    --batch_size 16 \
    --gradient_checkpointing \
    --adafactor \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --learning_rate 1e-5 \
    --exp_type $exp_type \
    --model_type $model_type \
    --model_name $model \
    --early_stopping 3 \
    --eval_baseline \
    --num_train_epochs 30 \
    &> $log_file
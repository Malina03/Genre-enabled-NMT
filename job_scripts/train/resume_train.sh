#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=24:00:00
#SBATCH --job-name=res
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

corpus=MaCoCu
language=$1 # the target language
exp_type=$2 # type of experiment (fine_tuned or from_scratch.)
model_type=$3 # type of model (genre_aware, genre_aware_token -genres are added as proper tokens- or baseline)
# genre=$5 # the genre that the model was trained on
use_tok=$4 # yes or no
use_old_data=$5 # yes or no
epochs=$6 # number of epochs to train for


echo "Use tokenizer: $use_tok"
echo "Use old data: $use_old_data"
echo "Model type: $model_type"
echo "Experiment type: $exp_type"
echo "Epochs: $epochs"

root_dir="/scratch/s3412768/genre_NMT/en-${language}"

if [ $language = 'hr' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi


if [ $use_old_data == 'yes' ]; then
    train_file="$root_dir/data/old_tokens/${corpus}.en-$language.train.tag.tsv"
    dev_file="${root_dir}/data/old_tokens/${corpus}.en-$language.dev.tag.tsv"
elif [ $use_old_data == 'no' ]; then
    if [ $model_type = 'genre_aware' ] || [ $model_type = 'genre_aware_token' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.train.tag.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.dev.tag.tsv"
    elif [ $model_type = 'doc_genre_aware' ] || [ $model_type = 'doc_genre_aware_token' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.doc.train.tag.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.doc.dev.tag.tsv"
    elif [ $model_type = 'baseline' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.train.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.dev.tsv"
    elif [ $model_type = 'doc_baseline' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.doc.train.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.doc.dev.tsv"
    else
        echo "Invalid model type"
        exit 1
    fi
else
    echo "Invalid use_old_data input"
    exit 1
fi

if [ $use_old_data == 'yes' ]; then
    model_type="od_${model_type}"
fi

if [ $use_tok == 'yes' ]; then
    model_type="tok_${model_type}"
fi

log_file="${root_dir}/logs/$exp_type/$model_type/train_${corpus}_2.log"
# if log directory does not exist, create it - but it really should exist
if [ ! -d "$root_dir/logs/$exp_type/$model_type/" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type/
fi

echo "Log file: $log_file"

checkpoint=$root_dir/models/from_scratch/$model_type/$corpus/checkpoint-*

echo "Checkpoint: $checkpoint"

if [ $use_tok == 'yes' ]; then 
    tokenizer_dir="$root_dir/models/from_scratch/$model_type/tokenizer"
    echo "Tokenizer: $tokenizer_dir"
    python /home1/s3412768/Genre-enabled-NMT/src/train.py \
        --root_dir $root_dir \
        --train_file $train_file \
        --dev_file $dev_file \
        --gradient_accumulation_steps 2 \
        --batch_size 16 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --learning_rate 1e-5 \
        --gradient_checkpointing \
        --adafactor \
        --wandb \
        --exp_type $exp_type \
        --model_type $model_type \
        --checkpoint $checkpoint \
        --model_name $model \
        --tokenizer_path $tokenizer_dir \
        --use_costum_tokenizer \
        --num_train_epochs $epochs \
        --early_stopping 10 \
        &> $log_file 
elif [ $use_tok == 'no' ]; then
    python /home1/s3412768/Genre-enabled-NMT/src/train.py \
        --root_dir $root_dir \
        --train_file $train_file \
        --dev_file $dev_file \
        --gradient_accumulation_steps 2 \
        --batch_size 16 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --learning_rate 1e-5 \
        --gradient_checkpointing \
        --adafactor \
        --wandb \
        --exp_type $exp_type \
        --model_type $model_type \
        --checkpoint $checkpoint \
        --model_name $model \
        --num_train_epochs $epochs \
        --early_stopping 10 \
        &> $log_file 
else
    echo "Invalid use_tok input"
    exit 1
fi
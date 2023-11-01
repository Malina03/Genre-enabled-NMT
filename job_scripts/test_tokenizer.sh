#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=02:00:00
#SBATCH --job-name=tok
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=164G


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
exp_type=$3 # type of model (e.g. fine_tuned or from_scratch.)
model_type=$4 # type of experiment ([doc_]genre_aware[_token] -genres are added as proper tokens- or [doc_]baseline)
genre=$5 # genre to fine-tune on 


root_dir="/scratch/s3412768/genre_NMT/en-$language"

if [ $language = 'hr' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi

if [ $exp_type = 'from_scratch' ]; then
    checkpoint=""
    genre=""
    log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/test_tokenizer_${corpus}.log"
    if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
        mkdir -p $root_dir/logs/$exp_type/$model_type
    fi

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

elif [ $exp_type = 'fine_tuned' ]; then
    checkpoint=$root_dir/models/from_scratch/$model_type/$corpus/checkpoint-*
    log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/$genre/train_${corpus}.log"
    if [ ! -d "$root_dir/logs/$exp_type/$model_type/$genre" ]; then
        mkdir -p $root_dir/logs/$exp_type/$model_type/$genre
    fi

    if [ $genre = 'doc' ]; then
        if [ $model_type = 'genre_aware' ] || [ $model_type = 'genre_aware_token' ]; then
            train_file="$root_dir/data/${corpus}.en-$language.doc.train.tag.tsv"
            dev_file="${root_dir}/data/${corpus}.en-$language.doc.dev.tag.tsv"
        elif [ $model_type = 'baseline' ]; then
            train_file="$root_dir/data/${corpus}.en-$language.doc.train.tsv"
            dev_file="${root_dir}/data/${corpus}.en-$language.doc.dev.tsv"
        else
            echo "Invalid model type"
            exit 1
        fi
    else
        if [ $model_type = 'genre_aware' ] || [ $model_type = 'genre_aware_token' ]; then
            train_file="$root_dir/data/${corpus}.en-$language.train.$genre.tag.tsv"
            dev_file="${root_dir}/data/${corpus}.en-$language.dev.$genre.tag.tsv"
        elif [ $model_type = 'baseline' ]; then
            train_file="$root_dir/data/${corpus}.en-$language.train.$genre.tsv"
            dev_file="${root_dir}/data/${corpus}.en-$language.dev.$genre.tsv"
        else
            echo "Invalid model type"
            exit 1
        fi
    fi
else
    echo "Invalid experiment type"
    exit 1
fi



echo "Checkpoint: $checkpoint"

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
    --num_train_epochs 20 \
    --train_tokenizer \
    &> $log_file
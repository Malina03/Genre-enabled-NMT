#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=24:00:00
#SBATCH --job-name=3_gen
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

corpus=$1 # corpus to fine-tune on
language=$2 # target language
exp_type=$3 # type of model (e.g. fine_tuned or from_scratch.)
model_type=$4 # type of experiment (tok_baseline, tok_genre_aware, tok_genre_aware_token)
genres=$5 # genre to fine-tune on 


root_dir="/scratch/s3412768/genre_NMT/en-$language"

if [ $language = 'hr' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi


if [ $exp_type = 'from_scratch' ]; then
    checkpoint=""
    genre=""
    if [ $model_type = 'genre_aware' ] || [ $model_type = 'genre_aware_token' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.train.$genres.tag.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.dev.$genres.tag.tsv"
    elif [ $model_type = 'baseline' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.train.$genres.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.dev.$genres.tsv"
    else
        echo "Invalid model type"
        exit 1
    fi
else
    echo "Invalid experiment type"
    exit 1
fi

## modify model type
model_type="tok_${genres}_${model_type}"

checkpoint=$root_dir/models/from_scratch/$model_type/$corpus/checkpoint-*
echo "Checkpoint: $checkpoint"

log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/train_${corpus}.log"
if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type
fi

# if train_file.src does not exist, create it
if [ ! -f "$train_file.src" ]; then
    echo "Creating $train_file.src"
    cut -f1 $train_file > $train_file.src
fi

# if train_file.ref does not exist, create it
if [ ! -f "$train_file.ref" ]; then
    echo "Creating $train_file.ref"
    cut -f2 $train_file > $train_file.ref
fi


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
    --num_train_epochs 20 \
    --early_stopping 20 \
    &> $log_file 
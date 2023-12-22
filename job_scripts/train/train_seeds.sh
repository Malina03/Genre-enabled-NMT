#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=72:00:00
#SBATCH --job-name=train_seeds
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --array=1-3

export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate


language=$1 # target language
model_type=$2 # type of experiment (baseline, genre_aware, genre_aware_token)
use_tok=$3 # yes or no

corpus="MaCoCu"
exp_type="from_scratch" # type of model (e.g. fine_tuned or from_scratch.)

root_dir="/scratch/s3412768/genre_NMT/en-$language"

echo "corpus: $corpus"
echo "language: $language"
echo "exp_type: $exp_type"
echo "model_type: $model_type"
echo "use_tok: $use_tok"

if [ $language = 'hr' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi

if [ $exp_type = 'from_scratch' ]; then
    if [ $model_type = 'genre_aware' ] || [ $model_type = 'genre_aware_token' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.train.tag.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.dev.tag.tsv"
    elif [ $model_type = 'baseline' ]; then
        train_file="$root_dir/data/${corpus}.en-$language.train.tsv"
        dev_file="${root_dir}/data/${corpus}.en-$language.dev.tsv"
    else
        echo "Invalid model type"
        exit 1
    fi
else
    echo "Invalid experiment type"
    exit 1
fi


echo "train file: $train_file"
echo "dev file: $dev_file"


## modify model type
if [ $use_tok == 'yes' ]; then
    model_type="tok_${model_type}"
    #if train_file.src does not exist, create it by cutting the first column of train_file
    if [ ! -f "$train_file.src" ]; then
        cut -f1 $train_file > $train_file.src
        cut -f2 $train_file > $train_file.ref
    fi
    #if dev_file.src does not exist, create it by cutting the first column of dev_file
    if [ ! -f "$dev_file.src" ]; then
        cut -f1 $dev_file > $dev_file.src
        cut -f2 $dev_file > $dev_file.ref
    fi

fi
# add seed to model type
model_type="${model_type}_${SLURM_ARRAY_TASK_ID}"
log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/train_${corpus}.log"
if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type
fi

echo "log file: $log_file"

if [ $use_tok == 'yes' ]; then
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
        --early_stopping 10 \
        --num_train_epochs 15 \
        --train_tokenizer \
        --seed $SLURM_ARRAY_TASK_ID \
        &> $log_file
else
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
        --early_stopping 10 \
        --num_train_epochs 15 \
        --seed $SLURM_ARRAY_TASK_ID \
        &> $log_file
fi
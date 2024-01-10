#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=3:00:00
#SBATCH --job-name=ft_seeds
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --array=1-6

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
seed=$4 # seed of the pretrained model

corpus="MaCoCu"
train_corpus="MaCoCu"
exp_type="fine_tune" # type of model (e.g. fine_tuned or from_scratch.)

root_dir="/scratch/s3412768/genre_NMT/en-$language"
genres=('news' 'law' 'arg' 'info' 'promo' 'random')
genre="${genres[$SLURM_ARRAY_TASK_ID-1]}"

# checkpoint=$root_dir/models/from_scratch/$model_type/$train_corpus/checkpoint-*


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

if [ $exp_type = 'fine_tune' ]; then
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
else
    echo "Invalid experiment type"
    exit 1
fi


echo "train file: $train_file"
echo "dev file: $dev_file"

if [ $use_tok == 'yes' ]; then
    tokenizer_path="/scratch/s3412768/genre_NMT/en-$language/models/from_scratch/tok_${model_type}_${seed}/tokenizer"
    echo "tokenizer path: $tokenizer_path"
    model_type="tok_${model_type}"
fi


checkpoint=$root_dir/models/from_scratch/${model_type}_${seed}/${train_corpus}/checkpoint-*


# add seed to model type
model_type="${model_type}_${genre}_${seed}"
log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/train.log"
if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type
fi

echo "log file: $log_file"

if [ $use_tok == 'yes' ]; then
    python /home1/s3412768/Genre-enabled-NMT/src/train.py \
        --root_dir $root_dir \
        --train_file $train_file \
        --dev_file $dev_file \
        --checkpoint $checkpoint \
        --wandb \
        --gradient_accumulation_steps 2 \
        --batch_size 16 \
        --gradient_checkpointing \
        --adafactor \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --learning_rate 1e-4 \
        --exp_type $exp_type \
        --model_type $model_type \
        --model_name $model \
        --early_stopping 5 \
        --num_train_epochs 5 \
        --use_costum_tokenizer \
        --tokenizer_path $tokenizer_path \
        --seed $seed \
        # --genre $genre \
        &> $log_file
else
    python /home1/s3412768/Genre-enabled-NMT/src/train.py \
        --root_dir $root_dir \
        --train_file $train_file \
        --dev_file $dev_file \
        --checkpoint $checkpoint \
        --wandb \
        --gradient_accumulation_steps 2 \
        --batch_size 16 \
        --gradient_checkpointing \
        --adafactor \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --learning_rate 1e-4 \
        --exp_type $exp_type \
        --model_type $model_type \
        --model_name $model \
        --early_stopping 5 \
        --num_train_epochs 5 \
        --seed $seed \
        # --genre $genre \
        &> $log_file
fi
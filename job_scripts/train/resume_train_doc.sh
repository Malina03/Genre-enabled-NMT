#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=06:00:00
#SBATCH --job-name=res_doc
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


language=$1 # target language
model_type=$2 # type of experiment (baseline, genre_aware, genre_aware_token)
seed=$3
epochs=$4
# genre=$5 # genre to fine-tune on 

corpus="MaCoCu"
# exp_type="from_scratch" # type of model (e.g. fine_tuned or from_scratch.)
exp_type="fine_tune"

root_dir="/scratch/s3412768/genre_NMT/en-$language"

if [ $language = 'hr' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
elif [ $language = 'tr' ]; then
    model="Helsinki-NLP/opus-mt-tc-big-en-tr"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi


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

if [ $language = 'hr' ]; then
    train_file_hr="${train_file}.hrv"
    if [[ ! -f $train_file_hr ]]; then
        echo "Train file for hr not found, create it"
        awk '{print ">>hrv<< " $0}' $train_file > $train_file_hr
    fi
    train_file=$train_file_hr
    
    dev_file_hr="${dev_file}.hrv"
    if [[ ! -f $dev_file_hr ]]; then
        echo "Dev file for hr not found, create it"
        awk '{print ">>hrv<< " $0}' $dev_file > $dev_file_hr
    fi
    dev_file=$dev_file_hr
fi
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
echo "train file: $train_file"
echo "dev file: $dev_file"

# use matching tokenizer from sentence level model

model_type="doc_${model_type}_${seed}"
checkpoint="$root_dir/models/$exp_type/$model_type/$corpus/checkpoint-*"

log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/train_${corpus}.log"
if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type
fi

echo "log file: $log_file"
echo "model type: $model_type"
echo "model: $model"
echo "exp_type: $exp_type"
echo "checkpoint: $checkpoint"

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
    --learning_rate 1e-4 \
    --exp_type $exp_type \
    --model_type $model_type \
    --model_name $model \
    --early_stopping 10 \
    --num_train_epochs $epochs \
    --checkpoint $checkpoint \
    --seed $seed \
    &> $log_file

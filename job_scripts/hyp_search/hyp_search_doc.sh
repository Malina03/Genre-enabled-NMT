#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=06:00:00
#SBATCH --job-name=hyp_search_doc
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=50G


export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

corpus="MaCoCu"
language="tr" # target language
exp_type="from_scratch/hyp_search" # type of model (e.g. fine_tuned or from_scratch.)
model_type="baseline" # type of experiment (baseline, genre_aware, genre_aware_token)
use_tok="no" # yes or no
lr=$1
bsz=$2
gac=$3




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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
echo "train file: $train_file"
echo "dev file: $dev_file"

# use matching tokenizer from sentence level model
# if [ $use_tok = 'yes' ]; then
#     tokenizer_path="/scratch/s3412768/genre_NMT/en-$language/models/from_scratch/tok_${model_type}_${SLURM_ARRAY_TASK_ID}/tokenizer"
#     echo "tokenizer path: $tokenizer_path"
#     model_type="tok_${model_type}"
# fi 

model_type="doc_${model_type}_${lr}_${bsz}_${gac}"


log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/train_${corpus}.log"
if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type
fi

echo "log file: $log_file"
echo "model type: $model_type"
echo "model: $model"


python /home1/s3412768/Genre-enabled-NMT/src/train.py \
    --root_dir $root_dir \
    --train_file $train_file \
    --dev_file $dev_file \
    --wandb \
    --gradient_accumulation_steps $gac \
    --batch_size $bsz \
    --learning_rate $lr \
    --gradient_checkpointing \
    --adafactor \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --exp_type $exp_type \
    --model_type $model_type \
    --model_name $model \
    --num_train_epochs 5 \
    &> $log_file

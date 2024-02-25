#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=03:00:00
#SBATCH --job-name=pred_seeds
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

train_corpus=MaCoCu
exp_type="from_scratch"
language=$1 # the target language
m_type=$2 # type of model (genre_aware, genre_aware_token -genres are added as proper tokens- or baseline)
# genre=$5 # the genre that the model was trained on
# test_on=$4 # the test file to evaluate on, assuming it is placed in root_dir/data
use_tok=$3 # yes or no


seed=$SLURM_ARRAY_TASK_ID

if [ $m_type == 'baseline' ]; then
    if [ $language == 'tr' ]; then
        test_files=("MaCoCu.en-${language}.test.tsv" "floresdev.en-${language}.test.tsv" "floresdevtest.en-${language}.test.tsv" "wmttest2018.en-${language}.test.tsv")
    elif [ $language == 'hr' ]; then
        test_files=("MaCoCu.en-${language}.test.tsv" "floresdev.en-${language}.test.tsv" "floresdevtest.en-${language}.test.tsv" "wmttest2022.en-${language}.test.tsv")
    elif [ $language == 'is' ]; then
        test_files=("MaCoCu.en-${language}.test.tsv" "floresdev.en-${language}.test.tsv" "floresdevtest.en-${language}.test.tsv" "wmttest2021.en-${language}.test.tsv")
    else
        echo "Invalid language"
        exit 1
    fi
elif [ $m_type == 'genre_aware' ] || [ $m_type == 'genre_aware_token' ]; then
    if [ $language == 'tr' ]; then
        test_files=("MaCoCu.en-${language}.test.tag.tsv" "floresdev.en-${language}.test.tag.tsv" "floresdevtest.en-${language}.test.tag.tsv" "wmttest2018.en-${language}.test.tag.tsv")
    elif [ $language == 'hr' ]; then
        test_files=("MaCoCu.en-${language}.test.tag.tsv" "floresdev.en-${language}.test.tag.tsv" "floresdevtest.en-${language}.test.tag.tsv" "wmttest2022.en-${language}.test.tag.tsv")
    elif [ $language == 'is' ]; then
        test_files=("MaCoCu.en-${language}.test.tag.tsv" "floresdev.en-${language}.test.tag.tsv" "floresdevtest.en-${language}.test.tag.tsv" "wmttest2021.en-${language}.test.tag.tsv")
    else
        echo "Invalid language"
        exit 1
    fi

else
    echo "Invalid model type"
    exit 1
fi

for test_on in "${test_files[@]}"; do

    module purge
    module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
    #load environment
    source /home1/s3412768/.envs/nmt2/bin/activate

    model_type=$m_type
    #
    if [ $use_tok == 'yes' ]; then
        model_type="tok_${model_type}"
    fi

    model_type="${model_type}_${seed}"

    echo "Use tokenizer: $use_tok"
    # echo "Use old data: $use_old_data"
    echo "Test on: $test_on"
    echo "Model type: $model_type"
    echo "Experiment type: $exp_type"

    root_dir="/scratch/s3412768/genre_NMT/en-${language}"

    if [ $language = 'hr' ]; then
        model="Helsinki-NLP/opus-mt-en-sla"
    elif 
        [ $language = 'tr' ]; then
        model="Helsinki-NLP/opus-mt-tc-big-en-tr"
    else
        model="Helsinki-NLP/opus-mt-en-${language}"
    fi

    test_file="${root_dir}/data/${test_on}"


    log_file="${root_dir}/logs/$exp_type/$model_type/eval_${test_on}.log"
    # if log directory does not exist, create it - but it really should exist
    if [ ! -d "$root_dir/logs/$exp_type/$model_type/" ]; then
        mkdir -p $root_dir/logs/$exp_type/$model_type/
    fi


    checkpoint=$root_dir/models/from_scratch/$model_type/$train_corpus/checkpoint-*
    echo "Checkpoint: $checkpoint"

    if [ $use_tok == 'yes' ]; then
        tokenizer_dir="$root_dir/models/from_scratch/$model_type/tokenizer"
        echo "Checkpoint: $checkpoint"
        echo "Tokenizer: $tokenizer_dir"
        python /home1/s3412768/Genre-enabled-NMT/src/train.py \
            --root_dir $root_dir \
            --train_file $test_file \
            --dev_file $test_file \
            --test_file $test_file\
            --gradient_accumulation_steps 2 \
            --batch_size 32 \
            --gradient_checkpointing \
            --adafactor \
            --exp_type $exp_type \
            --model_type $model_type \
            --checkpoint $checkpoint \
            --model_name $model \
            --tokenizer_path $tokenizer_dir \
            --use_costum_tokenizer \
            --eval \
            --predict \
            &> $log_file 
    elif [ $use_tok == 'no' ]; then
        python /home1/s3412768/Genre-enabled-NMT/src/train.py \
            --root_dir $root_dir \
            --train_file $test_file \
            --dev_file $test_file \
            --test_file $test_file\
            --gradient_accumulation_steps 2 \
            --batch_size 32 \
            --gradient_checkpointing \
            --adafactor \
            --exp_type $exp_type \
            --model_type $model_type \
            --checkpoint $checkpoint \
            --model_name $model \
            --eval \
            --predict \
            &> $log_file 
    else    
        echo "Use_tok should be yes or no"
        exit 1
    fi
        

    # deactivate the env used for predictions
    deactivate
    # remove the module used for predictions and load the new one
    module purge
    module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
    source $HOME/.envs/nmt_eval/bin/activate
    set -eu -o pipefail


    # Calculate all metrics between two files
    eval_file=$test_on
    out_file="$(cut -d'.' -f1 <<<"$test_on")"

    out=$root_dir/eval/$exp_type/$model_type/${out_file}_predictions.txt

    eval="$root_dir/data/${eval_file}"


    echo "Output file: $out"
    echo "Eval file: $eval"

    ref=${eval}.ref
    src=${eval}.src

    # check if ref and src files exist and create them if not
    if [[ ! -f $ref ]]; then
        echo "Reference file $ref not found, create it"
        # First check if the file exists in the data folder
        if [[ -f $eval ]]; then
            # If so, extract the reference column
            cut -d $'\t' -f2 $eval > "$ref"
        else
            echo "File $eval not found"
        fi
    fi

    if [[ ! -f $src ]]; then
        echo "Source file $src not found, create it"
        # First check if the file exists in the data folder
        if [[ -f $eval ]]; then
            # If so, extract the source column
            cut -d $'\t' -f1 $eval > "$src"
        else
            echo "File $eval not found"
        fi
    fi


    if [[ ! -f $out ]]; then
        echo "Output file $out not found, skip evaluation"
    else

        # First put everything in 1 file
        sacrebleu $out -i $ref -m bleu ter chrf --chrf-word-order 2 > "${out}.eval.sacre"
        # Add chrf++ to the previous file
        sacrebleu $out -i $ref -m chrf --chrf-word-order 2 >> "${out}.eval.sacre"
        # Write only scores to individual files
        sacrebleu $out -i $ref -m bleu -b > "${out}.eval.bleu"
        sacrebleu $out -i $ref -m ter -b > "${out}.eval.ter"
        sacrebleu $out -i $ref -m chrf -b > "${out}.eval.chrf"
        sacrebleu $out -i $ref -m chrf --chrf-word-order 2 -b > "${out}.eval.chrfpp"

        
        comet-score -s $src -t $out -r $ref > "${out}.eval.comet"

    fi

    # python /home1/s3412768/Genre-enabled-NMT/src/summarize.py \
    #     --folder $root_dir/eval/$exp_type/$model_type/ \
    #     --fname $out_file \
    #     --ref_with_tags $root_dir/data/$out_file.en-hr.test.tag.tsv \

done
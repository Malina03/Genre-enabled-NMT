#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=00:15:00
#SBATCH --job-name=prep
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=164G


export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


#load environment
source /home1/s3412768/.envs/nmt2/bin/activate

# root_dir="/scratch/s3412768/genre_NMT/en-hr/data"

# python -u /home1/s3412768/Genre-enabled-NMT/src/analyse.py > $root_dir/analyse.log
python -u /home1/s3412768/Genre-enabled-NMT/src/make_genre_datasets.py
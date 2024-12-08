#!/bin/bash 

#SBATCH -J fact_lrdg_APR.sh

#SBATCH -n 1 

#SBATCH --gres=gpu:V100:1

#SBATCH -o fact_lrdg_APR-%j.out

#SBATCH -e fact_lrdg_APR-%j.err

#SBATCH -t 14400

#SBATCH --mem=8000

module purge
module add nvidia/11.8
module add python/3.11
nvidia-smi

python3 train_lrdg.py --src Photos,Multispectral --trg APR

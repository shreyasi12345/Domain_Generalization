#!/bin/bash 

#SBATCH -J LRDG_DS_cartoon

#SBATCH -n 1 

#SBATCH --gres=gpu:V100:1

#SBATCH -o LRDG_DS_cartoon-%j.out

#SBATCH -e LRDG_DS_cartoon-%j.err

#SBATCH -t 14400

#SBATCH --mem=8000

module purge
module add nvidia/11.8
module add python/3.11
nvidia-smi
python3 /scratch/ghoshs/large_files/LRDG/train_ds.py --src cartoon,art,photo --trg sketch --batch-size 64
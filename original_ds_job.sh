#!/bin/bash 

#SBATCH -J Orginal_DS

#SBATCH -n 1 

#SBATCH --gres=gpu:A30:1

#SBATCH -o Orginal_DS-%j.out

#SBATCH -e Orginal_DS-%j.err

#SBATCH -t 14400

#SBATCH --mem=8000

module purge
module add nvidia/11.8
module add python/3.11
nvidia-smi
python3 /scratch/ghoshs/large_files/LRDG/original_train_ds.py --src art,photo,cartoon --trg sketch --batch-size 8

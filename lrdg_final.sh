#!/bin/bash 

#SBATCH -J lrdg_final.sh

#SBATCH -n 1 

#SBATCH --gres=gpu:V100:1

#SBATCH -o lrdg_final-%j.out

#SBATCH -e lrdg_final-%j.err

#SBATCH -t 14400

#SBATCH --mem=8000
#/scratch/ghoshs/large_files/LRDG/checkpoints/sketch_art,photo,cartoon_train_ds 

module purge
module add nvidia/11.8
module add python/3.11
nvidia-smi

python3 train_lrdg.py --src art,photo,cartoon --trg sketch --resume ./checkpoints/sketch_art,photo,cartoon_train_ds/00100.pth#./checkpoints/sketch_photo,cartoon,art_train_ds/00100.pth#./checkpoints/sketch_cartoon,art,photo_train_ds/00100.pth
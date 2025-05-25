#!/bin/bash 

#SBATCH -J lrdg_final_APR.sh

#SBATCH -n 1 

#SBATCH --gres=gpu:a30:1

#SBATCH -o lrdg_final_APR-%j.out

#SBATCH -e lrdg_final_APR-%j.err

#SBATCH -t 14400

#SBATCH --mem=8000


module purge
module add gcc/latest
module add nvidia/11.8
module add python/3.11
nvidia-smi

source /scratch/ghoshs/large_files/myenv/bin/activate

python3 /scratch/ghoshs/large_files/LRDG/train_lrdg.py --src Multispectral,Photos --trg APR --network resnet50 --class_list Tree --resume ./ds_multispctral/checkpoints/APR_Multispectral,Photos_train_ds/00100.pth#./ds_photos/checkpoints/APR_Photos,Multispectral_train_ds/00100.pth 

deactivate
module purge

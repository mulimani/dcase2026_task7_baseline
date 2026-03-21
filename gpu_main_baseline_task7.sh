#!/bin/bash
#SBATCH --job-name=task7_dil
#SBATCH --account=project_462001198
#SBATCH --output=out_task7_dil.txt
#SBATCH --error=err_task7_dil.txt
#SBATCH --partition=small-g
#SBATCH --time=02:30:00
#SBATCH --begin=now

#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gpus=1

module use /appl/local/csc/modulefiles/
module load pytorch
export NUMBA_CACHE_DIR="/scratch/project_462000765/manjunath/esc50_exp/"
export PYTHONUSERBASE="/scratch/project_462000765/manjunath/esc50_exp/"
export HF_HOME="/scratch/project_462000765/manjunath/esc50_exp/"

python baseline/baseline_DIL_task7.py train --augmentation='none' --learning_rate=1e-4 --batch_size=32 --cuda --num_workers 16 --epoch 120 --resume


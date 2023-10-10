#!/bin/bash
#SBATCH --job-name=n1
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=05:00:00
#SBATCH --reservation=ai_demo

export OMP_NUM_THREADS=1

module load gcc miniconda3
source $CONDA_PROFILE/conda.sh

conda activate /mnt/apps/custom/conda/envHPMLT

python __hpmlt__.py

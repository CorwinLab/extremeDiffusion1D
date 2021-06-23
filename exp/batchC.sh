#!/bin/bash
#SBATCH --job-name=generateDiffusionData
#SBATCH --time=1-00::00:00
#SBATCH --error=logs/generateDiffusionData.%A-%a.err
#SBATCH --notes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=logs/generateDiffusionData.%A-%a.out

TOPDIR=/home/jhass2/Data/
NUM=1e50
BETA=1

python3 /home/jhass2/code/extremeDiffusion1D/exp/getCdata.py $TOPDIR $NUM $BETA $SLURM_ARRAY_TASK_ID

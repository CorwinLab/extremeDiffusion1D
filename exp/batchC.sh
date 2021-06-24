#!/bin/bash
#SBATCH --job-name=generateDiffusionData
#SBATCH --time=0-00:03:00
#SBATCH --error=/home/jhass2/Data/log/generateDiffusionData.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5
#SBATCH --output=/home/jhass2/Data/log/generateDiffusionData.%A-%a.out

TOPDIR=/home/jhass2/Data/
NUM=1e10
BETA=1

python3 /home/jhass2/Code/extremeDiffusion1D/exp/getCdata.py $TOPDIR $NUM $BETA $SLURM_ARRAY_TASK_ID

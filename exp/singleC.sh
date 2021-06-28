#!/bin/bash
#SBATCH --job-name=generateDiffusionData
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/Data/log/generateDiffusionData.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1
#SBATCH --output=/home/jhass2/Data/log/generateDiffusionData.%A-%a.out

TOPDIR=/home/jhass2/Data/
NUM=1e300
BETA=1.0

python3 /home/jhass2/Code/extremeDiffusion1D/exp/getCdata.py $TOPDIR $NUM $BETA $SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --job-name=LargeQuartile
#SBATCH --time=24-00:00:00
#SBATCH --error=/home/jhass2/Data/log/LargeQuartile.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/Data/log/LargeQuartile.%A-%a.out

TOPDIR=/home/jhass2/Data/
python3 /home/jhass2/Code/extremeDiffusion1D/exp/NthQuartile/N300DataScript.py $TOPDIR $SLURM_ARRAY_TASK_ID

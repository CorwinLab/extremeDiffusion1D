#!/bin/bash
#SBATCH --job-name=PaperCDF
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Paper/CDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/CleanData/logs/Paper/CDF/%A-%a.out
#SBATCH --nice=1100

TOPDIR=/home/jhass2/CleanData/Paper/CDF
BETA=1
NUM_OF_SAVE_TIMES=7500

python3 ./CDFVarPaper.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $NUM_OF_SAVE_TIMES

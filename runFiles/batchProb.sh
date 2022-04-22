#!/bin/bash
#SBATCH --job-name=PaperMax
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Paper/Max/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=251-500
#SBATCH --output=/home/jhass2/CleanData/logs/Paper/Max/%A-%a.out
#SBATCH --nice=1000

TOP_DIR=./
BETA=1
QUANTILE=20
SLURM_ARRAY_TASK_ID=0

python3 ProbAndV.py $TOP_DIR $SLURM_ARRAY_TASK_ID $BETA $QUANTILE

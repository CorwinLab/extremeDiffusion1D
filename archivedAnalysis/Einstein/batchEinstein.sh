#!/bin/bash
#SBATCH --job-name=generateEinstein
#SBATCH --time=0-04:00:00
#SBATCH --error=/home/jhass2/Data/log/generateEinstein.%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/Data/log/generateEinstein.%A-%a.out

TOPDIR=/home/jhass2/Data/
NUM=1e50

python3 /home/jhass2/Code/extremeDiffusion1D/exp/getEinstein.py $TOPDIR $NUM $SLURM_ARRAY_TASK_ID

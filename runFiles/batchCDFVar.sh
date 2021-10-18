#!/bin/bash
#SBATCH --job-name=E100
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Einstein100/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH --output=/home/jhass2/CleanData/logs/Einstein100/%A-%a.out
#SBATCH --nice=1100

TOPDIR=/home/jhass2/CleanData/Einstein100
BETA=inf
TMAX=13000000
NUM_OF_SAVE_TIMES=7500
NParticles=300

python3 ./CDFVar.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $NParticles

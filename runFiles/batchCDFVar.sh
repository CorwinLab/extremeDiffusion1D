#!/bin/bash
#SBATCH --job-name=CDFVar100
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/CDFVar100/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10
#SBATCH --output=/home/jhass2/CleanData/logs/CDFVar100/%A-%a.out
#SBATCH --nice=1100

TOPDIR=/home/jhass2/CleanData/CDFVar100
BETA=1.0
TMAX=50000
NUM_OF_SAVE_TIMES=7500
NParticles=100

python3 ./CDFVar.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $NParticles

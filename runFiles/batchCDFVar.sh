#!/bin/bash
#SBATCH --job-name=CDFVar
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/CDFVarFixed/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/CleanData/logs/CDFVarFixed/%A-%a.out
#SBATCH --nice=1100

TOPDIR=/home/jhass2/CleanData/CDFVarFixed
BETA=1.0
TMAX=100000
NUM_OF_SAVE_TIMES=5000
NParticles=1000000

python3 ./CDFVar.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $NParticles

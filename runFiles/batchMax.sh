#!/bin/bash
#SBATCH --job-name=MaxBetaSweep
#SBATCH --time=2-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/MaxBetaSweep/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/CleanData/logs/MaxBetaSweep/%A-%a.out
#SBATCH --nice=1000

TOPDIR=/home/jhass2/CleanData/MaxBetaSweep/
NUM_OF_STEPS=300000
NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0
N_EXP=20

for BETA in 0.1 1 5 10
do
  python3 MaxPartDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $PROBDISTFLAG
done

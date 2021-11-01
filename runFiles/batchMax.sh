#!/bin/bash
#SBATCH --job-name=MaxPartSmall
#SBATCH --time=2-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/MaxPartSmall/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/CleanData/logs/MaxPartSmall/%A-%a.out
#SBATCH --nice=1000

TOPDIR=/home/jhass2/CleanData/MaxPartSmall/
BETA=1.0
NUM_OF_STEPS=100000
NUM_OF_SAVE_TIMES=7500
PROBDISTFLAG=0

for N_EXP in {2..10..1}
do
  python3 NthQuartileDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $PROBDISTFLAG
done

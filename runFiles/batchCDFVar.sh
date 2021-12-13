#!/bin/bash
#SBATCH --job-name=CDFSmallBeta1
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/CDFSmallBeta1/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-250
#SBATCH --output=/home/jhass2/CleanData/logs/CDFSmallBeta1/%A-%a.out
#SBATCH --nice=1100

TOPDIR=/home/jhass2/CleanData/CDFSmallBeta1
BETA=1
TMAX=300000
NUM_OF_SAVE_TIMES=7500
nStart=2
nStop=20
nStep=1

python3 ./CDFVar.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $nStart $nStop $nStep

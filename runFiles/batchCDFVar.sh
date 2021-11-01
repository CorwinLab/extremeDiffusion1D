#!/bin/bash
#SBATCH --job-name=ESweep
#SBATCH --time=10-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/EinsteinVariance/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH --output=/home/jhass2/CleanData/logs/EinstenVariance/%A-%a.out
#SBATCH --nice=1100

TOPDIR=/home/jhass2/CleanData/EinstenVariance
BETA=inf
TMAX=2000000
NUM_OF_SAVE_TIMES=7500
nStart=50
nStop=300
nStep=50

python3 ./CDFVar.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $nStart $nStop $nStep

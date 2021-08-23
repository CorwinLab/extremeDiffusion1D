#!/bin/bash
#SBATCH --job-name=QSweep
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/QSweep/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jhass2/CleanData/logs/QSweep/%A-%a.err
#SBATCH --nice=100

TOPDIR=/home/jhass2/CleanData/QSweep/
BETA=1.0
N_EXP=4500
NUM_OF_STEPS=300000
NUM_OF_SAVE_TIMES=5000
QUARTILE_START=10
QUARTILE_STOP=20
Q_STEP=1

python3 NthQuartileDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $Q_STEP

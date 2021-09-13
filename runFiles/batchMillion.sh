#!/bin/bash
#SBATCH --job-name=MaxPart300
#SBATCH --time=30-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/MaxPart300/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1000
#SBATCH --output=/home/jhass2/CleanData/logs/MaxPart300/%A-%a.out
#SBATCH --nice=1000

TOPDIR=/home/jhass2/CleanData/MaxPart300/
BETA=1.0
N_EXP=300
NUM_OF_STEPS=13000000
NUM_OF_SAVE_TIMES=7500
QUARTILE_START=20
QUARTILE_STOP=300
Q_STEP=20
PROBDISTFLAG=0

python3 NthQuartileDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $N_EXP $NUM_OF_STEPS $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $Q_STEP $PROBDISTFLAG

#!/bin/bash
#SBATCH --job-name=Recurrence
#SBATCH --time=20-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Recurrence/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5000%20
#SBATCH --output=/home/jhass2/CleanData/logs/Recurrence/%A-%a.err

TOPDIR=/home/jhass2/CleanData/Recurrence/
BETA=1.0
TMAX=3000000
NUM_OF_SAVE_TIMES=5000
QUARTILE_START=50
QUARTILE_STOP=4500
QUARTILE_STEP=50

python3 ./RecurrenceDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $QUARTILE_STEP

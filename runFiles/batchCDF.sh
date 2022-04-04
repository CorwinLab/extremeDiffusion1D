#!/bin/bash
#SBATCH --job-name=Recurrence
#SBATCH --time=20-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Recurrence/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/CleanData/logs/Recurrence/%A-%a.err
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/CleanData/Recurrence/
BETA=1.0
TMAX=100000
NUM_OF_SAVE_TIMES=7500
QUARTILE_START=2
QUARTILE_STOP=21
QUARTILE_STEP=1


mkdir -p $TOPDIR
python3 ./CDFDriver.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $TMAX $NUM_OF_SAVE_TIMES $QUARTILE_START $QUARTILE_STOP $QUARTILE_STEP

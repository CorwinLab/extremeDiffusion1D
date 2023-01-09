#!/bin/bash
#SBATCH --job-name=D25102
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Delta/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Delta/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
TOPDIR=/home/jhass2/jamming/JacobData/Delta/25102/
mkdir -p $TOPDIR

python3 ./DeltaCDF.py $TOPDIR $SLURM_ARRAY_TASK_ID $NUM_OF_SAVE_TIMES

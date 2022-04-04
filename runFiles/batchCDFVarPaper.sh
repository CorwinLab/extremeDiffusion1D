#!/bin/bash
#SBATCH --job-name=DiffCDF
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Paper/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Paper/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/Paper/
BETA=1
NUM_OF_SAVE_TIMES=7500

python3 ./CDFVarPaper.py $TOPDIR $SLURM_ARRAY_TASK_ID $BETA $NUM_OF_SAVE_TIMES

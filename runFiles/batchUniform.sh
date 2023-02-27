#!/bin/bash
#SBATCH --job-name=U84
#SBATCH --time=1-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Uniform/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Uniform/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
TOPDIR=/home/jhass2/jamming/JacobData/Uniform/84
mkdir -p $TOPDIR

python3 ./UniformCDF.py $TOPDIR $SLURM_ARRAY_TASK_ID $NUM_OF_SAVE_TIMES

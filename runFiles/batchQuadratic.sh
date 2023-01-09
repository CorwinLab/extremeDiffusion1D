#!/bin/bash
#SBATCH --job-name=Quad
#SBATCH --time=3-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/Quadratic/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/Quadratic/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

NUM_OF_SAVE_TIMES=7500
TOPDIR=/home/jhass2/jamming/JacobData/Quadratic/804
mkdir -p $TOPDIR

python3 ./QuadraticCDF.py $TOPDIR $SLURM_ARRAY_TASK_ID $NUM_OF_SAVE_TIMES

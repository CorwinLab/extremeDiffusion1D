#!/bin/bash
#SBATCH --job-name=BSweep
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/Bates/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/CleanData/logs/Bates/%A-%a.out
#SBATCH --nice=2000

NUM_OF_SAVE_TIMES=7500
N=7
TOPDIR=/home/jhass2/jamming/JacobData/Bates/$N
mkdir -p $TOPDIR

python3 ./UniformCDF.py $TOPDIR $SLURM_ARRAY_TASK_ID $N $NUM_OF_SAVE_TIMES
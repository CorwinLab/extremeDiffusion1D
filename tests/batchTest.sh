#!/bin/bash
#SBATCH --job-name=SaveTest
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/Test/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-20
#SBATCH --output=/home/jhass2/Test/%A-%a.out
#SBATCH --nice=1000

TOPDIR=/home/jhass2/Test/

python3 testCancel.py $TOPDIR $SLURM_ARRAY_TASK_ID

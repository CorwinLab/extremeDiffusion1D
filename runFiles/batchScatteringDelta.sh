#!/bin/bash
#SBATCH --job-name=SDelta
#SBATCH --time=0-06:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/ScatteringDelta/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/ScatteringDelta/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

TOPDIR=/home/jhass2/jamming/JacobData/ScatteringDeltaDist/
mkdir -p $TOPDIR

python3 scatteringQuantile.py $TOPDIR $SLURM_ARRAY_TASK_ID

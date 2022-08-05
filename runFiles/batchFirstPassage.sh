#!/bin/bash
#SBATCH --job-name=LCDF
#SBATCH --time=14-00:00:00
#SBATCH --error=/home/jhass2/CleanData/logs/LongFirstPassCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/CleanData/logs/LongFirstPassCDF/%A-%a.out
#SBATCH --nice=2000

BETA=1
NEXP=24
DMIN=10
DMAX=500
CUTOFF=1
TOPDIR=/home/jhass2/CleanData/LongFirstPassCDF/$NEXP

mkdir -p $TOPDIR

python3 FirstPassageTimes.py $TOPDIR $BETA $NEXP $SLURM_ARRAY_TASK_ID $DMIN $DMAX $CUTOFF

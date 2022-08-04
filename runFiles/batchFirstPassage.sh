#!/bin/bash
#SBATCH --job-name=LCDF
#SBATCH --time=14-00:00:00
#SBATCH --error=/home/jhass2/CleanData/LongFirstPassCDF/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-39
#SBATCH --output=/home/jhass2/CleanData/LongFirstPassCDF/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt

BETA=1
NEXP=24
DMIN=10
DMAX=500
CUTOFF=1
TOPDIR=home/jhass2/CleanData/LongFirstPassCDF/NEXP

mkdir -p $TOPDIR

for i in {0..12}
do
    id=$((SLURM_ARRAY_TASK_ID*12 + i + SLURM_ARRAY_TASK_ID))
    python3 FirstPassageTimes.py $TOPDIR $BETA $NEXP $id $DMIN $DMAX $CUTOFF
done

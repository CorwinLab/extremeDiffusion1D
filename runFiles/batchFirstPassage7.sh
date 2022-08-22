#!/bin/bash
#SBATCH --job-name=2LCDF
#SBATCH --time=14-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/LongFirstPassCDF2/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-9
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/LongFirstPassCDF2/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=long
#SBATCH --requeue

BETA=1
NEXP=2
DMIN=10
DMAX=500
CUTOFF=1
TOPDIR=/home/jhass2/jamming/JacobData/LongFirstPassCDF2/

mkdir -p $TOPDIR

for i in {0..49}
do
	id=$((SLURM_ARRAY_TASK_ID*50 + i + SLURM_ARRAY_TASK_ID))
	python3 FirstPassageTimes.py $TOPDIR $BETA $NEXP $id $DMIN $DMAX $CUTOFF
done

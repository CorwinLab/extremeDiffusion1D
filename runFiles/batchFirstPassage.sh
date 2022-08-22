#!/bin/bash
#SBATCH --job-name=24LCDF
#SBATCH --time=7-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/LongFirstPassCDF24/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-499
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/LongFirstPassCDF24/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

BETA=1
NEXP=24
DMIN=10
DMAX=500
CUTOFF=1
TOPDIR=/home/jhass2/jamming/JacobData/LongFirstPassCDF24/

mkdir -p $TOPDIR

python3 FirstPassageTimes.py $TOPDIR $BETA $NEXP $SLURM_ARRAY_TASK_ID $DMIN $DMAX $CUTOFF
